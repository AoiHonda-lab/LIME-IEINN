# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.segmentation import mark_boundaries
# from skimage.transform import resize
# import matplotlib.patches as patches

# class Explanation:
#     def __init__(self, local_exp, intercept, score, local_pred,
#                  segments=None, original_image=None, grid_size=None):
#         self.local_exp = local_exp
#         self.intercept = intercept
#         self.score = score
#         self.local_pred = local_pred
#         self.segments = segments
#         self.original_image = original_image
#         self.grid_size = grid_size

#     def plot_image_explanation(
#         self,
#         label=0,
#         positive_only=False,
#         negative_only=False,
#         hide_rest=False,
#         num_features=10,
#         min_weight=0.0,
#         title=None
#     ):
#         """
#         LIMEのオリジナルロジックに基づく可視化関数。
#         正負の重みによってRGBの特定チャンネルを強調することで色分け。

#         Args:
#             label (int): 説明対象ラベル（通常は1クラス分類なので0）
#             positive_only (bool): 正の貢献の特徴だけを可視化
#             negative_only (bool): 負の貢献の特徴だけを可視化
#             hide_rest (bool): 非注目セグメントをグレーアウト
#             num_features (int): 上位の特徴数
#             min_weight (float): 最小重要度しきい値
#             title (str): 表示用タイトル
#         """
#         if positive_only and negative_only:
#             raise ValueError("positive_only and negative_only cannot both be True.")
#         if self.segments is None or self.original_image is None:
#             raise ValueError("segments and original_image are required for image explanation.")

#         segments = self.segments
#         image = self.original_image
#         exp = self.local_exp  # list of (feature_idx, weight)
        
#         # マスクとテンポラリ画像の初期化
#         mask = np.zeros(segments.shape, dtype=int)
#         if hide_rest:
#             temp = np.zeros_like(image)
#         else:
#             temp = image.copy()

#         if positive_only:
#             selected = [i for i, w in exp if w > min_weight][:num_features]
#             for f in selected:
#                 temp[segments == f] = image[segments == f]
#                 mask[segments == f] = 1

#         elif negative_only:
#             selected = [i for i, w in exp if w < -min_weight][:num_features]
#             for f in selected:
#                 temp[segments == f] = image[segments == f]
#                 mask[segments == f] = -1

#         else:
#             # 両方表示
#             for f, w in exp[:num_features]:
#                 if abs(w) < min_weight:
#                     continue
#                 c = 0 if w < 0 else 1  # R if negative, G if positive
#                 mask[segments == f] = -1 if w < 0 else 1
#                 temp[segments == f] = image[segments == f].copy()
#                 temp[segments == f, c] = np.max(image)  # 強調色（明るく）

#         vis_image = temp  # LIME風の明るさ補正
#         plt.figure(figsize=(8, 8))
#         plt.imshow(mark_boundaries(vis_image, mask))
#         plt.title(title or "LIME Explanation (Green=Positive, Red=Negative)")
#         plt.axis("off")
#         plt.tight_layout()
#         plt.show()

#     def plot_segment_grid_view(self, grid_size=4, title=None):
#         if self.segments is None or self.original_image is None:
#             raise ValueError("segments and original_image are required for image explanation.")

#         segments = self.segments
#         image = self.original_image
#         h, w = segments.shape

#         # 各領域のセグメント数（想定：連番）
#         unique_segments = np.unique(segments)
#         n_segs = len(unique_segments)

#         n_rows = n_cols = grid_size  # 例：4×4

#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
#         fig.suptitle(title or f"{grid_size}x{grid_size} Grid of Segments", fontsize=16)

#         for idx, seg_id in enumerate(unique_segments):
#             mask = segments == seg_id
#             row = idx // grid_size
#             col = idx % grid_size

#             ax = axes[row, col]
#             seg_img = np.zeros_like(image)
#             seg_img[mask] = image[mask]  # 対象領域のみ表示（他は黒）

#             ax.imshow(seg_img)
#             ax.set_title(f"Segment {seg_id}", fontsize=10)
#             ax.axis("off")

#         plt.tight_layout()
#         plt.subplots_adjust(top=0.92)  # タイトルスペース
#         plt.show()

#     def plot_segment_tiles_with_margin(self, grid_size=4, margin=5, title=None):
#         if self.segments is None or self.original_image is None:
#             raise ValueError("segments and original image are required for image explanation.")

#         image = self.original_image
#         segments = self.segments
#         h, w = segments.shape

#         tile_h = h // grid_size
#         tile_w = w // grid_size

#         canvas_h = grid_size * (tile_h + margin) - margin
#         canvas_w = grid_size * (tile_w + margin) - margin
#         canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255  # 白背景

#         seg_ids = np.unique(segments)

#         for seg_id in seg_ids:
#             mask = segments == seg_id
#             seg_img = np.zeros_like(image)
#             seg_img[mask] = image[mask]

#             # セグメント領域を bounding box で切り出し
#             coords = np.argwhere(mask)
#             y_min, x_min = coords.min(axis=0)
#             y_max, x_max = coords.max(axis=0) + 1

#             seg_crop = seg_img[y_min:y_max, x_min:x_max]
#             seg_resized = resize(seg_crop, (tile_h, tile_w), preserve_range=True).astype(np.uint8)

#             row = seg_id // grid_size
#             col = seg_id % grid_size
#             y0 = row * (tile_h + margin)
#             x0 = col * (tile_w + margin)

#             canvas[y0:y0 + tile_h, x0:x0 + tile_w] = seg_resized

#         plt.figure(figsize=(10, 10))
#         plt.imshow(canvas)
#         plt.title(title or f"{grid_size}x{grid_size} Segments View")
#         plt.axis("off")
#         plt.tight_layout()
#         plt.show()

#     def plot_colored_segment_tiles(self, top_k=10, min_weight=0.0, title=None):
#         if self.segments is None or self.original_image is None:
#             raise ValueError("segments and original_image are required for image explanation.")

#         image = self.original_image
#         segments = self.segments
#         grid_size = self.grid_size
#         h, w = segments.shape

#         tile_h = h // grid_size
#         tile_w = w // grid_size
#         margin = 10

#         canvas_h = grid_size * (tile_h + margin) - margin
#         canvas_w = grid_size * (tile_w + margin) - margin
#         canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255  # 白背景
#         overlay = np.zeros_like(canvas, dtype=float)

#         exp = sorted(self.local_exp, key=lambda x: abs(x[1]), reverse=True)
#         exp = [x for x in exp if abs(x[1]) >= min_weight][:top_k]
#         highlighted_ids = {seg_id for seg_id, _ in exp}

#         seg_ids = np.unique(segments)

#         for seg_id in seg_ids:
#             mask = segments == seg_id
#             seg_img = np.zeros_like(image)
#             seg_img[mask] = image[mask]

#             coords = np.argwhere(mask)
#             y_min, x_min = coords.min(axis=0)
#             y_max, x_max = coords.max(axis=0) + 1
#             seg_crop = seg_img[y_min:y_max, x_min:x_max]
#             seg_resized = resize(seg_crop, (tile_h, tile_w), preserve_range=True).astype(np.uint8)

#             row = seg_id // grid_size
#             col = seg_id % grid_size
#             y0 = row * (tile_h + margin)
#             x0 = col * (tile_w + margin)

#             canvas[y0:y0 + tile_h, x0:x0 + tile_w] = seg_resized

#             if seg_id in highlighted_ids:
#                 weight = dict(exp)[seg_id]
#                 color = np.array([255, 0, 0]) if weight < 0 else np.array([0, 255, 0])  # 赤 or 緑
#                 color_layer = np.ones((tile_h, tile_w, 3)) * color / 255.0
#                 overlay[y0:y0 + tile_h, x0:x0 + tile_w] = color_layer

#         canvas_normalized = canvas / 255.0
#         canvas_corrected = canvas_normalized #/ 2 + 0.5
#         blended = canvas_corrected + overlay * 0.5

#         fig, ax = plt.subplots(figsize=(10, 10))
#         ax.imshow(blended)
#         ax.set_title(title or "Colored Explanation over All Segments")
#         ax.axis("off")

#         for seg_id, weight in exp:
#             row = seg_id // grid_size
#             col = seg_id % grid_size
#             y0 = row * (tile_h + margin)
#             x0 = col * (tile_w + margin)

#             edge_color = 'green' if weight > 0 else 'red'
#             rect = patches.Rectangle(
#                 (x0, y0), tile_w, tile_h,
#                 linewidth=2, edgecolor=edge_color, facecolor='none'
#             )
#             ax.add_patch(rect)

#         plt.tight_layout()
#         plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

class Explanation:
    def __init__(self, local_exp, intercept, score, local_pred,
                 segments=None, original_image=None, grid_size=4,
                 shapley=None, fuzzy=None):
        self.local_exp = local_exp  # list of (subset, weight) - mobius interaction weights
        self.intercept = intercept
        self.score = score
        self.local_pred = local_pred
        self.segments = segments
        self.original_image = original_image
        self.grid_size = grid_size
        self.shapley = shapley  # dict: feature_index -> shapley value
        self.fuzzy = fuzzy      # dict: subset(tuple) -> fuzzy value

    def plot_colored_segment_tiles(self, top_k=10, min_weight=0.0, title=None):
        if self.segments is None or self.original_image is None or self.shapley is None:
            raise ValueError("segments, original image, and shapley values are required for image explanation.")

        image = self.original_image
        segments = self.segments
        grid_size = self.grid_size
        h, w = segments.shape

        tile_h = h // grid_size
        tile_w = w // grid_size
        margin = 10

        canvas_h = grid_size * (tile_h + margin) - margin
        canvas_w = grid_size * (tile_w + margin) - margin
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255  # 白背景
        overlay = np.zeros_like(canvas, dtype=float)

        # shapley辞書をarrayに変換
        if isinstance(self.shapley, dict):
            shapley_array = np.zeros(max(self.shapley.keys()) + 1)
            for i, v in self.shapley.items():
                shapley_array[i] = v
        else:
            shapley_array = self.shapley

        exp_indices = np.argsort(np.abs(shapley_array))[-top_k:]
        highlighted_ids = set(exp_indices)

        seg_ids = np.unique(segments)

        for seg_id in seg_ids:
            mask = segments == seg_id
            seg_img = np.zeros_like(image)
            seg_img[mask] = image[mask]

            coords = np.argwhere(mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1
            seg_crop = seg_img[y_min:y_max, x_min:x_max]
            seg_resized = resize(seg_crop, (tile_h, tile_w), preserve_range=True).astype(np.uint8)

            row = seg_id // grid_size
            col = seg_id % grid_size
            y0 = row * (tile_h + margin)
            x0 = col * (tile_w + margin)

            canvas[y0:y0 + tile_h, x0:x0 + tile_w] = seg_resized

            if seg_id in highlighted_ids:
                weight = shapley_array[seg_id]
                color = np.array([255, 0, 0]) if weight < 0 else np.array([0, 255, 0])  # 赤 or 緑
                color_layer = np.ones((tile_h, tile_w, 3)) * color / 255.0
                overlay[y0:y0 + tile_h, x0:x0 + tile_w] = color_layer

        canvas_normalized = canvas / 255.0
        canvas_corrected = canvas_normalized
        blended = canvas_corrected + overlay * 0.5

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(blended)
        ax.set_title(title or "Colored Explanation over All Segments (Shapley)")
        ax.axis("off")

        for seg_id in highlighted_ids:
            row = seg_id // grid_size
            col = seg_id % grid_size
            y0 = row * (tile_h + margin)
            x0 = col * (tile_w + margin)
            edge_color = 'green' if shapley_array[seg_id] > 0 else 'red'
            rect = patches.Rectangle((x0, y0), tile_w, tile_h,
                                     linewidth=2, edgecolor=edge_color, facecolor='none')
            ax.add_patch(rect)

        plt.tight_layout()
        plt.show()

    def plot(self, mode="shapley", top_k=10):
        if mode == "shapley" and self.shapley is not None:
            values = self.shapley if not isinstance(self.shapley, dict) else np.array([
                self.shapley[k] for k in sorted(self.shapley.keys())
            ])
            title = "Shapley Values"
            labels = list(sorted(self.shapley.keys())) if isinstance(self.shapley, dict) else list(range(len(values)))
        elif mode == "mobius" and self.local_exp is not None:
            values = np.array([w for _, w in self.local_exp])
            title = "Mobius (Interaction) Values"
            labels = ["∩".join(map(str, subset)) for subset, _ in self.local_exp]
        elif mode == "fuzzy" and self.fuzzy is not None:
            values = np.array([w for _, w in self.fuzzy.items()])
            title = "Fuzzy Values"
            labels = ["∩".join(map(str, subset)) for subset in self.fuzzy.keys()]
        else:
            raise ValueError("Invalid mode or data not available for: " + mode)

        top_k = min(top_k, len(values))
        indices = np.argsort(np.abs(values))[-top_k:]

        plt.figure(figsize=(10, 4))
        plt.bar(range(top_k), values[indices], tick_label=[labels[i] for i in indices])
        plt.title(title)
        plt.xlabel("Feature / Subset")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    # def plot_circular_interaction_graph(self, norm=True, figsize=(10, 10)):
    #     if self.segments is None or self.original_image is None:
    #         raise ValueError("original_image and segments must be set.")

    #     image = self.original_image
    #     segments = self.segments
    #     shapley = self.shapley or {}
    #     local_exp = self.local_exp or []

    #     grid_size = self.grid_size
    #     tile_h = image.shape[0] // grid_size
    #     tile_w = image.shape[1] // grid_size

    #     fig, ax = plt.subplots(figsize=figsize)

    #     center_x, center_y = image.shape[1] / 2, image.shape[0] / 2

    #     # 1. 分割タイルを中央付近に再配置（少しずらして）
    #     margin = 10
    #     scale = 0.5
    #     tile_w_scaled = int(tile_w * scale)
    #     tile_h_scaled = int(tile_h * scale)

    #     total_w = grid_size * tile_w_scaled + (grid_size - 1) * margin
    #     total_h = grid_size * tile_h_scaled + (grid_size - 1) * margin

    #     offset_x = center_x - total_w / 2
    #     offset_y = center_y - total_h / 2

    #     tile_positions = {}
    #     tile_idx = 0
    #     for i in range(grid_size):
    #         for j in range(grid_size):
    #             x0 = j * tile_w
    #             y0 = i * tile_h
    #             tile = image[y0:y0+tile_h, x0:x0+tile_w]
    #             pos_x = offset_x + j * (tile_w_scaled + margin)
    #             pos_y = offset_y + i * (tile_h_scaled + margin)
    #             ax.imshow(tile, extent=[pos_x, pos_x + tile_w_scaled, pos_y, pos_y + tile_h_scaled], zorder=3)
    #             tile_positions[tile_idx] = (
    #                 pos_x + tile_w_scaled / 2,
    #                 pos_y + tile_h_scaled / 2
    #             )
    #             tile_idx += 1

    #      # 2. セグメントごとの中心点と外周配置座標
    #     unique_segments = sorted(np.unique(segments))
    #     n = len(unique_segments)
    #     radius = max(image.shape) * 0.75  # 外周半径
    #     inner_radius = max(image.shape) * 0.6  # 線と円の内周半径
    #     center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
    #     outer_positions = {}
    #     inner_positions = {}
    #     segment_centers = {}

    #     outer_scale = 0.6  # 分割画像サイズ縮小係数

    #     for idx, seg_id in enumerate(unique_segments):
    #         coords = np.argwhere(segments == seg_id)
    #         y_mean, x_mean = coords.mean(axis=0)
    #         segment_centers[seg_id] = (x_mean, y_mean)

    #         angle = 2 * np.pi * idx / n
    #         cx_outer = center_x + radius * np.cos(angle)
    #         cy_outer = center_y + radius * np.sin(angle)
    #         outer_positions[seg_id] = (cx_outer, cy_outer)

    #         cx_inner = center_x + inner_radius * np.cos(angle)
    #         cy_inner = center_y + inner_radius * np.sin(angle)
    #         inner_positions[seg_id] = (cx_inner, cy_inner)

    #         # トリミングして貼り付け（外側）
    #         seg_img = np.zeros_like(image)
    #         seg_img[segments == seg_id] = image[segments == seg_id]
    #         y_min, x_min = coords.min(axis=0)
    #         y_max, x_max = coords.max(axis=0) + 1
    #         crop = resize(seg_img[y_min:y_max, x_min:x_max], (int(tile_h * outer_scale), int(tile_w * outer_scale)), preserve_range=True).astype(np.uint8)
    #         ax.imshow(crop, extent=[cx_outer - tile_w*outer_scale/2, cx_outer + tile_w*outer_scale/2, cy_outer - tile_h*outer_scale/2, cy_outer + tile_h*outer_scale/2], zorder=2)

    #     # 3. 相互作用（local_exp）を線で描画（内周）
    #     if local_exp:
    #         all_weights = [abs(w) for (k, w) in local_exp if len(k) == 2]
    #         max_weight = max(all_weights) if norm and all_weights else 1
    #         for (seg_pair, weight) in local_exp:
    #             if len(seg_pair) != 2: continue
    #             i, j = seg_pair
    #             if i not in inner_positions or j not in inner_positions:
    #                 continue
    #             x0, y0 = inner_positions[i]
    #             x1, y1 = inner_positions[j]
    #             color = 'blue' if weight > 0 else 'red'
    #             lw = 0.5 + 4 * abs(weight) / max_weight
    #             ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=0.6, zorder=1)

    #     # 4. Shapley値を円で描画（内周）
    #     shapley_vals = list(map(abs, shapley.values()))
    #     max_shap = max(shapley_vals) if norm and shapley_vals else 1
    #     for seg_id, val in shapley.items():
    #         if seg_id not in inner_positions: continue
    #         cx, cy = inner_positions[seg_id]
    #         r = 5 + 25 * abs(val) / max_shap
    #         circle = patches.Circle((cx, cy), radius=r, color='red', alpha=0.5, zorder=3)
    #         ax.add_patch(circle)

    #     # 5. 凡例を追加
    #     legend_elements = [
    #         Line2D([0], [0], color='red', lw=2, label='Negative Interaction'),
    #         Line2D([0], [0], color='blue', lw=2, label='Positive Interaction'),
    #         patches.Circle((0, 0), radius=10, color='red', alpha=0.5, label='Shapley (Importance)')
    #     ]
    #     ax.legend(handles=legend_elements, loc='upper right')

    #     ax.set_title("n-SII values of order k = 1, 2\nInteraction Graph (Shapley + Mobius)", fontsize=14)
    #     ax.axis("off")
    #     plt.tight_layout()
    #     plt.show()

    def plot_circular_interaction_graph(self, norm=True, figsize=(10, 10)):
        if self.segments is None or self.original_image is None:
            raise ValueError("original_image and segments must be set.")

        image = self.original_image
        segments = self.segments
        shapley = self.shapley or {}
        local_exp = self.local_exp or []

        grid_size = self.grid_size
        tile_h = image.shape[0] // grid_size
        tile_w = image.shape[1] // grid_size

        fig, ax = plt.subplots(figsize=figsize)

        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2

        # === [修正] === scale の調整
        scale = 0.5
        tile_h_scaled = int(tile_h * scale)
        tile_w_scaled = int(tile_w * scale)

        # 1. 分割タイルを中央付近に再配置
        margin = 10
        total_w = grid_size * tile_w_scaled + (grid_size - 1) * margin
        total_h = grid_size * tile_h_scaled + (grid_size - 1) * margin
        offset_x = center_x - total_w / 2
        offset_y = center_y - total_h / 2

        flipped_image = np.flipud(self.original_image)
        tile_positions = {}
        tile_idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                y0 = row * tile_h
                x0 = col * tile_w
                tile = flipped_image[y0:y0+tile_h, x0:x0+tile_w]
                pos_x = offset_x + col * (tile_w_scaled + margin)
                pos_y = offset_y + row * (tile_h_scaled + margin)
                ax.imshow(tile, extent=[pos_x, pos_x + tile_w_scaled, pos_y + tile_h_scaled, pos_y], zorder=3)
                rect = patches.Rectangle((pos_x, pos_y), tile_w_scaled, tile_h_scaled,
                                        linewidth=1, edgecolor='black', facecolor='none', zorder=3)
                ax.add_patch(rect)
                tile_positions[tile_idx] = (pos_x + tile_w_scaled / 2, pos_y + tile_h_scaled / 2)
                tile_idx += 1

        # 2. セグメントごとの中心点と外周配置座標
        unique_segments = sorted(np.unique(segments))
        n = len(unique_segments)
        radius = max(image.shape) * 0.75  # 外周半径
        inner_radius = max(image.shape) * 0.6  # 線と円の内周半径
        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
        outer_positions = {}
        inner_positions = {}
        segment_centers = {}

        outer_scale = 0.6  # 分割画像サイズ縮小係数

        for idx, seg_id in enumerate(unique_segments):
            coords = np.argwhere(segments == seg_id)
            y_mean, x_mean = coords.mean(axis=0)
            segment_centers[seg_id] = (x_mean, y_mean)

            angle = 2 * np.pi * idx / n
            cx_outer = center_x + radius * np.cos(angle)
            cy_outer = center_y + radius * np.sin(angle)
            outer_positions[seg_id] = (cx_outer, cy_outer)

            cx_inner = center_x + inner_radius * np.cos(angle)
            cy_inner = center_y + inner_radius * np.sin(angle)
            inner_positions[seg_id] = (cx_inner, cy_inner)

            # トリミングして貼り付け（外側）
            seg_img = np.zeros_like(image)
            seg_img[segments == seg_id] = image[segments == seg_id]
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1
            crop = resize(seg_img[y_min:y_max, x_min:x_max], (int(tile_h * outer_scale), int(tile_w * outer_scale)), preserve_range=True).astype(np.uint8)
            ax.imshow(crop, extent=[cx_outer - tile_w*outer_scale/2, cx_outer + tile_w*outer_scale/2, cy_outer - tile_h*outer_scale/2, cy_outer + tile_h*outer_scale/2], zorder=2)
            rect = patches.Rectangle((cx_outer - tile_w*outer_scale/2, cy_outer - tile_h*outer_scale/2),
                                    tile_w*outer_scale, tile_h*outer_scale,
                                    linewidth=1, edgecolor='black', facecolor='none', zorder=3)
            ax.add_patch(rect)

        # 3. 相互作用（local_exp）を線で描画（内周）
        if local_exp:
            all_weights = [abs(w) for (k, w) in local_exp if len(k) == 2]
            max_weight = max(all_weights) if norm and all_weights else 1
            for (seg_pair, weight) in local_exp:
                if len(seg_pair) != 2: continue
                i, j = seg_pair
                if i not in inner_positions or j not in inner_positions:
                    continue
                x0, y0 = inner_positions[i]
                x1, y1 = inner_positions[j]
                color = 'blue' if weight > 0 else 'red'
                lw = 0.5 + 4 * abs(weight) / max_weight
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=0.6, zorder=1)

        # 4. Shapley値を円で描画（内周）
        shapley_vals = list(map(abs, shapley.values()))
        max_shap = max(shapley_vals) if norm and shapley_vals else 1
        for seg_id, val in shapley.items():
            if seg_id not in inner_positions: continue
            cx, cy = inner_positions[seg_id]
            r = 5 + 25 * abs(val) / max_shap
            circle = patches.Circle((cx, cy), radius=r, color='red', alpha=0.5, zorder=4)
            ax.add_patch(circle)

        # 5. 凡例を追加
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Negative Interaction'),
            Line2D([0], [0], color='blue', lw=2, label='Positive Interaction'),
            patches.Circle((0, 0), radius=10, color='red', alpha=0.5, label='Shapley (Importance)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title("n-SII values of order k = 1, 2\nInteraction Graph (Shapley + Mobius)", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.show()