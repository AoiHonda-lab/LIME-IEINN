import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Explanation:
    def __init__(self, local_exp, intercept, score, local_pred,
                 segments=None, original_image=None, grid_size=4,
                 shapley=None, fuzzy=None, interaction=None, feature_names=None):
        self.local_exp = local_exp  # list of (subset, weight) - mobius interaction weights
        self.intercept = intercept
        self.score = score
        self.local_pred = local_pred
        self.segments = segments
        self.original_image = original_image
        self.grid_size = grid_size
        self.shapley = shapley  # dict: feature_index -> shapley value
        self.fuzzy = fuzzy      # dict: subset(tuple) -> fuzzy value
        self.interaction = interaction 
        self.feature_names = feature_names

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



    def plot(self, mode="shapley", top_k=10, min_scale=0.5, max_scale=1.0, show_segments=True):
        if self.local_exp is None:
            raise ValueError("local_exp must be set for bar chart.")
        if show_segments and (self.segments is None or self.original_image is None or self.shapley is None):
            raise ValueError("segments, original_image, and shapley must be set if show_segments=True.")
        if mode == "fuzzy" and self.fuzzy is None:
            raise ValueError("fuzzy mode requires fuzzy values.")
        if mode == "interaction" and self.interaction is None:
            raise ValueError("interaction mode requires interaction values.")
        if mode == "shapley" and self.shapley is None:
            raise ValueError("shapley mode requires shapley values.")

        # shapleyの整形（abs）
        if isinstance(self.shapley, dict):
            shapley_array = np.zeros(max(self.shapley.keys()) + 1)
            for i, v in self.shapley.items():
                shapley_array[i] = abs(v)
        else:
            shapley_array = np.abs(self.shapley)
        max_shap = np.max(shapley_array) if np.max(shapley_array) > 0 else 1e-5

        # 描画対象の値
        if mode == "shapley":
            bar_vals = list(enumerate(self.shapley if not isinstance(self.shapley, dict)
                                    else [self.shapley.get(i, 0) for i in range(len(shapley_array))]))
        elif mode == "mobius":
            bar_vals = self.local_exp
        elif mode == "fuzzy":
            bar_vals = list(self.fuzzy.items())
        elif mode == "interaction":
            bar_vals = list(self.interaction.items())
        else:
            raise ValueError(f"Invalid mode: {mode}")

        bar_vals = sorted(bar_vals, key=lambda x: abs(x[1]), reverse=True)[:top_k]
        labels = ["∩".join(map(str, subset)) if isinstance(subset, (list, tuple, set)) else str(subset)
                for subset, _ in bar_vals]
        values = [v for _, v in bar_vals]
        colors = ['green' if v > 0 else 'red' for v in values]

        # 描画準備
        if show_segments:
            segments = self.segments
            image = self.original_image
            grid_size = self.grid_size
            h, w = segments.shape
            tile_h = h // grid_size
            tile_w = w // grid_size
            margin = 10
            canvas_h = grid_size * (tile_h + margin) - margin
            canvas_w = grid_size * (tile_w + margin) - margin

            fig, (ax_img, ax_bar) = plt.subplots(
                2, 1, figsize=(10, 10), height_ratios=[3, 1], gridspec_kw={'hspace': 0.4}
            )

            for seg_id in np.unique(segments):
                mask = segments == seg_id
                seg_img = np.zeros_like(image)
                seg_img[mask] = image[mask]

                coords = np.argwhere(mask)
                if coords.size == 0:
                    continue
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0) + 1
                seg_crop = seg_img[y_min:y_max, x_min:x_max]

                shap_val = shapley_array[seg_id] if seg_id < len(shapley_array) else 0.0
                scale = min_scale + (shap_val / max_shap) * (max_scale - min_scale)
                scaled_h = int(tile_h * scale)
                scaled_w = int(tile_w * scale)

                seg_resized = resize(seg_crop, (scaled_h, scaled_w), preserve_range=True).astype(np.uint8)

                row = seg_id // grid_size
                col = seg_id % grid_size
                y0 = row * (tile_h + margin) + (tile_h - scaled_h) // 2
                x0 = col * (tile_w + margin) + (tile_w - scaled_w) // 2

                ax_img.imshow(seg_resized, extent=[x0, x0 + scaled_w, y0 + scaled_h, y0], zorder=1)

                # 黒枠
                rect = patches.Rectangle((x0, y0), scaled_w, scaled_h,
                                        linewidth=1.5, edgecolor='black', facecolor='none', zorder=2)
                ax_img.add_patch(rect)

                # --- 改良されたセグメント番号の描写 ---
                ax_img.text(
                    x0 + scaled_w / 2, y0 + scaled_h / 2,
                    str(seg_id),
                    ha='center', va='center',
                    fontsize=15,
                    weight='bold',
                    color='black',
                    path_effects=[
                        path_effects.Stroke(linewidth=2, foreground='white'),
                        path_effects.Normal()
                    ],
                    zorder=3
                )

            ax_img.set_xlim(0, canvas_w)
            ax_img.set_ylim(canvas_h, 0)
            ax_img.set_title(f"Tiles Scaled by Shapley Value", fontsize=14)
            ax_img.axis("off")
        else:
            fig, ax_bar = plt.subplots(figsize=(10, 4))

        # 棒グラフの描画
        ax = ax_bar
        ax.bar(range(len(values)), values, tick_label=labels, color=colors)
        ax.set_title(f"{mode.capitalize()} Values (Top {top_k})", fontsize=14)
        ax.set_ylabel("Value")
        yticklabels = ax.get_yticklabels()
        ax.set_yticklabels(yticklabels, fontsize=13)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=13)
        ax.axhline(0, color='black', linewidth=1)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        plt.subplots_adjust(hspace=0.4)
        plt.show()


    def plot_circular_interaction_graph(self, norm=True, figsize=(10, 10)):
        if self.segments is None or self.original_image is None:
            raise ValueError("original_image and segments must be set.")

        image = self.original_image
        segments = self.segments
        shapley = self.shapley or {}
        #local_exp = self.local_exp or []
        interaction = list(self.interaction.items())

        grid_size = self.grid_size
        tile_h = image.shape[0] // grid_size
        tile_w = image.shape[1] // grid_size

        fig, ax = plt.subplots(figsize=figsize)
        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2

        # === 1. 中央配置（固定スケール） ===
        scale = 0.5
        tile_h_scaled = int(tile_h * scale)
        tile_w_scaled = int(tile_w * scale)
        margin = 10
        total_w = grid_size * tile_w_scaled + (grid_size - 1) * margin
        total_h = grid_size * tile_h_scaled + (grid_size - 1) * margin
        offset_x = center_x - total_w / 2
        offset_y = center_y - total_h / 2

        flipped_image = np.flipud(image)
        tile_idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                y0 = row * tile_h
                x0 = col * tile_w
                tile = flipped_image[y0:y0 + tile_h, x0:x0 + tile_w]
                pos_x = offset_x + col * (tile_w_scaled + margin)
                pos_y = offset_y + row * (tile_h_scaled + margin)
                ax.imshow(tile, extent=[pos_x, pos_x + tile_w_scaled, pos_y + tile_h_scaled, pos_y], zorder=3)
                rect = patches.Rectangle((pos_x, pos_y), tile_w_scaled, tile_h_scaled,
                                        linewidth=1, edgecolor='black', facecolor='none', zorder=3)
                ax.add_patch(rect)
                tile_idx += 1

        # === 2. 外周配置（Shapleyに基づくスケール） ===
        unique_segments = sorted(np.unique(segments))
        n = len(unique_segments)
        radius = max(image.shape) * 0.75
        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2

        if isinstance(shapley, dict):
            shapley_array = np.zeros(max(shapley.keys()) + 1)
            for i, v in shapley.items():
                shapley_array[i] = abs(v)
        else:
            shapley_array = np.abs(shapley)

        max_shap = np.max(shapley_array) if np.max(shapley_array) > 0 else 1e-5
        outer_scale_min, outer_scale_max = 0.3, 0.8

        for idx, seg_id in enumerate(unique_segments):
            coords = np.argwhere(segments == seg_id)
            angle = 2 * np.pi * idx / n
            cx_outer = center_x + radius * np.cos(angle)
            cy_outer = center_y + radius * np.sin(angle)

            # Shapleyスケーリング
            shap_val = shapley_array[seg_id] if seg_id < len(shapley_array) else 0
            scale = outer_scale_min + (shap_val / max_shap) * (outer_scale_max - outer_scale_min)
            th = int(tile_h * scale)
            tw = int(tile_w * scale)

            seg_img = np.zeros_like(image)
            seg_img[segments == seg_id] = image[segments == seg_id]
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1
            crop = resize(seg_img[y_min:y_max, x_min:x_max], (th, tw), preserve_range=True).astype(np.uint8)

            ax.imshow(crop,
                    extent=[cx_outer - tw / 2, cx_outer + tw / 2,
                            cy_outer - th / 2, cy_outer + th / 2],
                    zorder=2)
            rect = patches.Rectangle((cx_outer - tw / 2, cy_outer - th / 2), tw, th,
                                    linewidth=1, edgecolor='black', facecolor='none', zorder=3)
            ax.add_patch(rect)

        # === 3. Interaction Lines (Mobius) ===
        inner_radius = max(image.shape) * 0.6
        inner_positions = {
            seg_id: (
                center_x + inner_radius * np.cos(2 * np.pi * idx / n),
                center_y + inner_radius * np.sin(2 * np.pi * idx / n)
            ) for idx, seg_id in enumerate(unique_segments)
        }

        #if interaction:
        if interaction:
            all_weights = [abs(w) for (k, w) in interaction if len(k) == 2]
            max_weight = max(all_weights) if norm and all_weights else 1
            for (seg_pair, weight) in interaction:
                if len(seg_pair) != 2:
                    continue
                i, j = seg_pair
                if i not in inner_positions or j not in inner_positions:
                    continue
                x0, y0 = inner_positions[i]
                x1, y1 = inner_positions[j]
                color = 'green' if weight > 0 else 'red'
                lw = 0.3 + 5 * abs(weight) / max_weight
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=0.6, zorder=1)

        # === 4. Mobius円（1次項） ===
        singleton_mobius = {k[0]: v for (k, v) in interaction if isinstance(k, (list, tuple)) and len(k) == 1}
        mobius_vals = list(map(abs, singleton_mobius.values()))
        max_mob = max(mobius_vals) if norm and mobius_vals else 1

        for seg_id, val in singleton_mobius.items():
            if seg_id not in inner_positions:
                continue
            cx, cy = inner_positions[seg_id]
            r = 5 + 25 * abs(val) / max_mob
            color = 'green' if val > 0 else 'red'
            ax.add_patch(patches.Circle((cx, cy), radius=r, color=color, alpha=0.5, zorder=4))

        # 凡例
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Negative Interaction'),
            Line2D([0], [0], color='green', lw=2, label='Positive Interaction'),
            patches.Circle((0, 0), radius=10, color='green', alpha=0.5, label='Interaction (Singleton +)'),
            patches.Circle((0, 0), radius=10, color='red', alpha=0.5, label='Interaction (Singleton -)'),
            Line2D([0], [0], color='black', lw=0, label='Outer Tile Size ∝ |Shapley|')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.set_title(
            "n-SII values of order k = 1, 2\n"
            "Interaction Graph with Outer Tiles Scaled by |Shapley|",
            fontsize=13
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig('circular.SVG')
        plt.show()

    def plot_top_mobius_tiles_scaled_by_shapley_safe(self, min_scale=0.5, max_scale=1.0, title=None):
        if self.segments is None or self.original_image is None or self.local_exp is None or self.shapley is None:
            raise ValueError("segments, original image, local_exp, and shapley are required.")

        image = self.original_image
        segments = self.segments
        grid_size = self.grid_size
        h, w = segments.shape

        tile_h = h // grid_size
        tile_w = w // grid_size
        margin = 10

        canvas_h = grid_size * (tile_h + margin) - margin
        canvas_w = grid_size * (tile_w + margin) - margin
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        overlay = np.zeros_like(canvas, dtype=float)

        # Mobius: 最大の組み合わせと符号
        top_subset, top_value = max(self.local_exp, key=lambda x: abs(x[1]))
        highlight_ids = set(top_subset)
        highlight_color = np.array([0, 255, 0]) if top_value > 0 else np.array([255, 0, 0])

        # Shapleyの準備
        if isinstance(self.shapley, dict):
            shapley_array = np.zeros(max(self.shapley.keys()) + 1)
            for i, v in self.shapley.items():
                shapley_array[i] = abs(v)
        else:
            shapley_array = np.abs(self.shapley)

        max_shap = np.max(shapley_array) if np.max(shapley_array) > 0 else 1e-5

        fig, ax = plt.subplots(figsize=(6, 6))

        for seg_id in np.unique(segments):
            # 元画像の該当セグメント抽出
            mask = segments == seg_id
            seg_img = np.zeros_like(image)
            seg_img[mask] = image[mask]

            coords = np.argwhere(mask)
            if coords.size == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1
            seg_crop = seg_img[y_min:y_max, x_min:x_max]

            # Shapley値 → スケール
            shap_val = shapley_array[seg_id] if seg_id < len(shapley_array) else 0.0
            scale = min_scale + (shap_val / max_shap) * (max_scale - min_scale)
            scaled_h = int(tile_h * scale)
            scaled_w = int(tile_w * scale)

            seg_resized = resize(seg_crop, (scaled_h, scaled_w), preserve_range=True).astype(np.uint8)

            # タイル内の配置
            row = seg_id // grid_size
            col = seg_id % grid_size
            y0 = row * (tile_h + margin) + (tile_h - scaled_h) // 2
            x0 = col * (tile_w + margin) + (tile_w - scaled_w) // 2

            # 安全にキャンバスに描画
            canvas[y0:y0 + scaled_h, x0:x0 + scaled_w] = seg_resized

            # ハイライト色塗り
            if seg_id in highlight_ids:
                color_layer = np.ones((scaled_h, scaled_w, 3)) * highlight_color / 255.0
                overlay[y0:y0 + scaled_h, x0:x0 + scaled_w] = color_layer

            # 全領域に黒枠を追加
            rect = patches.Rectangle((x0, y0), scaled_w, scaled_h,
                                    linewidth=1.5, edgecolor='black', facecolor='none', zorder=2)
            ax.add_patch(rect)

        # 画像合成と表示
        canvas_normalized = canvas / 255.0
        blended = np.clip(canvas_normalized + overlay * 0.5, 0, 1)

        ax.imshow(blended)
        ax.set_title(title or f"Top Mobius Subset: {top_subset} (value={top_value:.3f})")
        ax.axis("off")
        plt.tight_layout()
        plt.show()



    def plot_multi_mobius_layers_stacked(self, num_layers=3, 
                                      layer_min_scale=0.6, layer_max_scale=1.3,
                                      tile_min_scale=0.8, tile_max_scale=1.2,
                                      offset_step=40, title=None):

        if self.segments is None or self.original_image is None or self.local_exp is None or self.shapley is None:
            raise ValueError("segments, original_image, local_exp, and shapley are required.")

        image = self.original_image
        segments = self.segments
        grid_size = self.grid_size
        h, w = segments.shape
        tile_h = h // grid_size
        tile_w = w // grid_size
        margin = 10
        canvas_h = grid_size * (tile_h + margin) - margin + offset_step * (num_layers - 1)
        canvas_w = grid_size * (tile_w + margin) - margin + offset_step * (num_layers - 1)

        if isinstance(self.shapley, dict):
            shapley_array = np.zeros(max(self.shapley.keys()) + 1)
            for i, v in self.shapley.items():
                shapley_array[i] = abs(v)
        else:
            shapley_array = np.abs(self.shapley)
        max_shap = np.max(shapley_array) if np.max(shapley_array) > 0 else 1e-5

        sorted_mobius = sorted(self.local_exp, key=lambda x: abs(x[1]), reverse=True)[:num_layers]
        max_mobius_val = abs(sorted_mobius[0][1]) if sorted_mobius else 1e-5

        fig, ax = plt.subplots(figsize=(6, 6))

        for layer_idx, (subset, mobius_val) in enumerate(sorted_mobius):
            # Mobiusスケール
            layer_scale = layer_min_scale + (abs(mobius_val) / max_mobius_val) * (layer_max_scale - layer_min_scale)
            highlight_ids = set(subset)
            highlight_color = np.array([0, 255, 0]) if mobius_val > 0 else np.array([255, 0, 0])
            edge_color = 'green' if mobius_val > 0 else 'red'

            offset_x = layer_idx * offset_step
            offset_y = layer_idx * offset_step

            for seg_id in np.unique(segments):
                mask = segments == seg_id
                seg_img = np.zeros_like(image)
                seg_img[mask] = image[mask]

                coords = np.argwhere(mask)
                if coords.size == 0:
                    continue

                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0) + 1
                seg_crop = seg_img[y_min:y_max, x_min:x_max]

                # Shapleyスケール
                shap_val = shapley_array[seg_id] if seg_id < len(shapley_array) else 0.0
                tile_scale = tile_min_scale + (shap_val / max_shap) * (tile_max_scale - tile_min_scale)

                total_scale = layer_scale * tile_scale
                scaled_h = int(tile_h * total_scale)
                scaled_w = int(tile_w * total_scale)
                seg_resized = resize(seg_crop, (scaled_h, scaled_w), preserve_range=True).astype(np.uint8)

                row = seg_id // grid_size
                col = seg_id % grid_size
                y0 = row * (tile_h + margin) + (tile_h - scaled_h) // 2 + offset_y
                x0 = col * (tile_w + margin) + (tile_w - scaled_w) // 2 + offset_x

                x0 = np.clip(x0, 0, canvas_w - scaled_w)
                y0 = np.clip(y0, 0, canvas_h - scaled_h)

                ax.imshow(seg_resized, extent=[x0, x0 + scaled_w, y0 + scaled_h, y0],
                        zorder=layer_idx * 10 + 1, alpha=0.95)

                if seg_id in highlight_ids:
                    overlay_color = np.ones((scaled_h, scaled_w, 3)) * highlight_color / 255.0
                    ax.imshow(overlay_color, extent=[x0, x0 + scaled_w, y0 + scaled_h, y0],
                            zorder=layer_idx * 10 + 2, alpha=0.4)

                ax.add_patch(patches.Rectangle((x0, y0), scaled_w, scaled_h,
                                            linewidth=1.5, edgecolor='black', facecolor='none',
                                            zorder=layer_idx * 10 + 3))
                ax.add_patch(patches.Rectangle((x0, y0), scaled_w, scaled_h,
                                            linewidth=1, edgecolor=edge_color, facecolor='none',
                                            zorder=layer_idx * 10 + 4))

        ax.set_xlim(0, canvas_w)
        ax.set_ylim(canvas_h, 0)
        ax.set_title(title or f"Top {num_layers} Mobius Layers (scaled & stacked)")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def plot_interaction_3d_aligned(self, top_k=5):
        if self.segments is None or self.original_image is None or self.local_exp is None:
            raise ValueError("segments, original_image, and local_exp must be set.")
        grid_size = self.grid_size
        tile_size = 1
        margin = 0.2
        fixed_spacing = 2.0
        image_plane_y = 0.5

        image = self.original_image
        segments = self.segments
        mobius_terms_sorted = sorted(self.local_exp, key=lambda x: abs(x[1]), reverse=True)[:top_k]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

        grid_total_width = grid_size * (tile_size + margin)
        grid_total_height = grid_size * (tile_size + margin)
        max_val = max(abs(v) for _, v in mobius_terms_sorted)
        bar_base_z = -grid_total_height

        # --- 分割画像（Y=0.5） ---
        for seg_id in np.unique(segments):
            mask = segments == seg_id
            seg_img = np.zeros_like(image)
            seg_img[mask] = image[mask]

            coords = np.argwhere(mask)
            if coords.size == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1
            seg_crop = seg_img[y_min:y_max, x_min:x_max]
            seg_resized = resize(seg_crop, (int(tile_size * 50), int(tile_size * 50)), preserve_range=True) / 255.0

            row, col = divmod(seg_id, grid_size)
            x0 = col * (tile_size + margin)
            z0 = -row * (tile_size + margin)

            xx, zz = np.meshgrid(
                np.linspace(x0, x0 + tile_size, seg_resized.shape[1]),
                np.linspace(z0, z0 - tile_size, seg_resized.shape[0])
            )
            yy = np.ones_like(xx) * image_plane_y

            ax.plot_surface(xx, yy, zz,
                facecolors=seg_resized,
                alpha=1.0,
                shade=False,
                zorder=0)

            # 黒枠の追加
            corners = [
                (x0, z0),
                (x0 + tile_size, z0),
                (x0 + tile_size, z0 - tile_size),
                (x0, z0 - tile_size),
                (x0, z0)
            ]
            for i in range(4):
                x_start, z_start = corners[i]
                x_end, z_end = corners[i + 1]
                ax.plot([x_start, x_end], [image_plane_y]*2, [z_start, z_end],
                        color='black', linewidth=1.0, zorder=1)

        # --- 棒グラフ ---
        for i, (subset, value) in enumerate(mobius_terms_sorted):
            color = 'green' if value > 0 else 'red'
            y = 0 if i == 0 else -i * fixed_spacing
            height = (abs(value) / max_val) * grid_total_width
            z = bar_base_z
            ax.plot([0, 0], [y, y], [z, z + height],
                    color=color, linewidth=8, solid_capstyle='round', zorder=2)

        # --- 多角形 + 点 + 点線 ---
        for i, (subset, value) in enumerate(mobius_terms_sorted):
            color = 'green' if value > 0 else 'red'
            y = 0 if i == 0 else -i * fixed_spacing
            xs, zs = [], []
            for idx in subset:
                col, row = divmod(idx, grid_size)
                x = col * (tile_size + margin) + tile_size / 2
                z = -row * (tile_size + margin) - tile_size / 2
                xs.append(x)
                zs.append(z)

                # 点線：Y方向（画像へ）/ X方向（棒グラフへ）
                ax.plot([x, x], [y, image_plane_y], [z, z],
                        linestyle='dotted', color='dimgray', linewidth=1.5, zorder=3)
                ax.plot([x, 0], [y, y], [z, z],
                        linestyle='dotted', color='dimgray', linewidth=1.5, zorder=3)
                  
                ax.plot([0, 0], [y, y], [bar_base_z, 0.5],
                color='black', linestyle='-', linewidth=1.0, zorder=1)

            # 多角形と点の描画
            if len(xs) == 1:
                ax.plot([xs[0]], [y], [zs[0]], marker='o', color=color, markersize=6, zorder=5)
            elif len(xs) == 2:
                ax.plot(xs, [y]*2, zs, color=color, linewidth=2, zorder=4)
                for x, z in zip(xs, zs):
                    ax.plot([x], [y], [z], marker='o', color=color, markersize=6, zorder=5)
            else:
                xs.append(xs[0])
                zs.append(zs[0])
                ax.plot(xs, [y]*len(xs), zs, color=color, linewidth=2, zorder=4)
                for x, z in zip(xs[:-1], zs[:-1]):
                    ax.plot([x], [y], [z], marker='o', color=color, markersize=6, zorder=5)

        # --- 軸範囲・スタイル ---
        ax.set_xlim(0, grid_total_width)
        ax.set_ylim(-fixed_spacing * (top_k - 1) - 1, 1)
        ax.set_zlim(bar_base_z - 0.5, 0.5)

        ax.set_ylabel("Interaction Rank Descending")

        # 数値目盛を消す
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(False)

        ax.set_title(f"Interaction 3D Layout (top {top_k})", fontsize=14)
        ax.view_init(elev=30, azim=-45)
        plt.subplots_adjust(hspace=0.4)
        plt.show()