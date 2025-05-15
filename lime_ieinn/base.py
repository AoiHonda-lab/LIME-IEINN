import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import torch
import torch.nn as nn
from .ieinn.ieinn import IE
from .pytorchtools import EarlyStopping
from .explanation import Explanation
from itertools import combinations
import pandas as pd

class LocalExplainer:
    def __init__(self, kernel_fn, additivity_order=1, epochs=100):
        self.kernel_fn = kernel_fn
        self.additivity_order = additivity_order
        self.epochs = epochs

    def explain(self, data, labels, distances, label=0, num_features=5):
        weights = self.kernel_fn(distances)
        y = labels[:, label]

        # Ridge回帰で特徴量選択
        model = Ridge(alpha=1.0)
        model.fit(data, y, sample_weight=weights)
        coefs = model.coef_
        top_features = np.argsort(np.abs(coefs))[-num_features:]

        # 特徴符号補正付きのデータ作成
        X = data[:, top_features]
        y = y.reshape(-1, 1)
        weights = weights.reshape(-1, 1)

        corr = pd.DataFrame(np.hstack((X, y))).corr()
        X_corr = X.copy()
        flag_corr = []
        for i in range(X.shape[1]):
            if corr.iloc[X.shape[1], i] < 0:
                X_corr[:, i] = -X_corr[:, i] + 1
                flag_corr.append(0)
            else:
                flag_corr.append(1)

        # デバイス設定
        device = 'cpu'
        torch.set_default_device(device)

        # データをtensor化してtrain/testに分割
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X_corr, y, weights, test_size=0.2, random_state=42
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        w_train = torch.tensor(w_train, dtype=torch.float32)
        w_test = torch.tensor(w_test, dtype=torch.float32)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train, w_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test, w_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        # IEINN モデル学習
        easy_model = IE(train_loader, additivity_order=self.additivity_order, narray_op='Algebraic', preprocessing='PreprocessingLayerNone').to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(easy_model.parameters(), lr=0.1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3)
        # early_stopping = EarlyStopping(patience=20, verbose=0)

        easy_model.fit_and_valid(train_loader, test_loader, criterion, optimizer,
                                 epochs=self.epochs, regularization=True, mono_lambda=0.2)

        # 予測と出力
        x0 = torch.tensor(X_corr[0].reshape(1, -1), dtype=torch.float32)
        local_pred = easy_model.predict(x0).item()

        # サブセット（特徴の組み合わせ）生成
        subsets = []
        for i in range(1, len(top_features) + 1):
            if i > self.additivity_order:
                break
            for c in combinations(top_features, i):
                subsets.append(c)

        # モデル出力
        weights_out = list(easy_model.output.weight[0].detach().cpu().numpy())
        fuzzy_out = list(easy_model.mobius_to_fuzzy().values())[1:]
        interaction_out = list(easy_model.interaction_indices().values())[1:]
        shapley_dict_local = easy_model.shapley()  # ローカルインデックスのShapley値

        local_exp = list(zip(subsets, weights_out))
        fuzzy = dict(zip(subsets, fuzzy_out))
        interaction = dict(zip(subsets, interaction_out))
        shapley = {top_features[key[0] - 1]: value for key, value in shapley_dict_local.items() if len(key) == 1}

        return Explanation(
            local_exp=local_exp,
            intercept=0.0,
            score=None,
            local_pred=local_pred,
            shapley=shapley,
            fuzzy=fuzzy,
            interaction=interaction,
        )
