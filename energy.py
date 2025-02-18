import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

def compute_eps(x, surf_energy, density):
    return (((6.0 * surf_energy) / density) ** 3 * ((0.000001 * x) ** (-5))) ** 0.5


models = []
best_model = None


def reset_session_state():
    """Удаляет ключи, связанные с обучением и данными, чтобы полностью сбросить состояние."""
    keys_to_clear = ["df", "X", "y", "df_show", "feature_cols",
                     "surf_energy", "density", "models_list", 
                     "trained", "training_results", "best_model"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def show_menu():
    st.title("Регрессионные модели для оценки удельной мощности")
    st.write(
        """
        Загрузите CSV файл с данными.
        Обязательно должна быть колонка **d** – конечный размер.
        Остальные колонки используются как входные признаки.
        """
    )

    # Загрузка файла с ключом для сохранения в session_state
    uploaded_file = st.file_uploader("Выберите CSV файл", type="csv", key="uploaded_file")
    if uploaded_file is None:
        return

    # Если загружен новый файл (по имени), сбрасываем предыдущие данные
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = uploaded_file.name
    elif st.session_state.uploaded_filename != uploaded_file.name:
        reset_session_state()
        st.session_state.uploaded_filename = uploaded_file.name

    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Загруженный набор данных:")
        st.dataframe(df.head())
        st.session_state["df"] = df
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        return

    if "d" not in df.columns:
        st.error("В наборе данных отсутствует обязательная колонка **d**.")
        return

    # Выбор параметров для расчёта eps
    surf_energy = st.slider("Поверхностная энергия", min_value=0.0001, max_value=100.0, value=0.15, step=0.001)
    density = st.slider("Плотность", min_value=1.0, max_value=50000.0, value=4000.0, step=100.0)
    st.session_state["surf_energy"] = surf_energy
    st.session_state["density"] = density

    try:
        # Вычисляем eps по формуле с правильно расставленными скобками
        df["eps"] = df["d"].apply(lambda x: (((6.0 * surf_energy) / density) ** 3 * ((0.000001 * x) ** (-5))) ** 0.5)
    except Exception as e:
        st.error(f"Ошибка при вычислении eps: {e}")
        return

    df_show = df.copy()  # копия для отображения результатов

    # Определяем признаки для обучения: все столбцы, кроме "d" и "eps"
    feature_cols = [col for col in df.columns if col not in ["d", "eps"]]
    if len(feature_cols) == 0:
        st.warning("В наборе данных нет дополнительных признаков. Добавляем фиктивный признак.")
        df["dummy"] = 1.0
        feature_cols = ["dummy"]

    st.write("#### Признаки для обучения:", feature_cols)
    X = df[feature_cols]
    y = df["eps"]

    # Сохраняем данные в session_state
    st.session_state["X"] = X
    st.session_state["y"] = y
    st.session_state["df_show"] = df_show
    st.session_state["feature_cols"] = feature_cols

    # Если моделей ещё не создано – инициализируем их
    if "models_list" not in st.session_state:
        mlp_regressor = MLPRegressor(hidden_layer_sizes=(10, 5),
                                     activation="relu",
                                     solver="lbfgs",
                                     learning_rate="invscaling",
                                     learning_rate_init=0.001,
                                     alpha=0.0001,
                                     max_iter=10000000)
        models_list = [
            Pipeline([("scaler", MinMaxScaler()), ("model", mlp_regressor)]),
            Pipeline([("scaler", MinMaxScaler()), ("model", SVR(kernel="rbf", C=100, gamma=50.0, epsilon=0.0001))]),
            Pipeline([("scaler", MinMaxScaler()), ("model", RandomForestRegressor(n_estimators=50, max_features="sqrt"))]),
            Pipeline([("scaler", MinMaxScaler()), ("model", KNeighborsRegressor(n_neighbors=4))]),
            Pipeline([("scaler", MinMaxScaler()), ("model", DecisionTreeRegressor())]),
            Pipeline([("scaler", MinMaxScaler()), ("model", GradientBoostingRegressor())])
        ]
        st.session_state["models_list"] = models_list

    # Флаг, показывающий, что модели уже обучены
    if "trained" not in st.session_state:
        st.session_state["trained"] = False

    # Обучение моделей
    if st.button("Обучить модели") or st.session_state["trained"]:
        if not st.session_state["trained"]:
            results = []
            for model in st.session_state["models_list"]:
                model_name = model.named_steps["model"].__class__.__name__
                st.write(f"**Обучение модели: {model_name}**")
                try:
                    model.fit(X, y)
                    y_pred_train = model.predict(X)
                    r2_val = r2_score(y, y_pred_train)
                    mse_val = mean_squared_error(y, y_pred_train)
                    results.append({"Model": model_name,
                                    "R2_eps": r2_val,
                                    "MSE": mse_val,
                                    "Model_obj": model})
                    st.write(f"{model_name}: R² = {r2_val:.4f}, MSE = {mse_val:.4f}")
                except Exception as e:
                    st.error(f"Ошибка при обучении модели {model_name}: {e}")
            st.session_state["training_results"] = results
            st.session_state["trained"] = True
        else:
            results = st.session_state["training_results"]

        if results:
            results_df = pd.DataFrame(results).set_index("Model")
            st.write("### Результаты обучения моделей:")
            st.dataframe(results_df[["R2_eps", "MSE"]])

            # Выбор модели для визуализации
            model_options = [r["Model"] for r in results]
            selected_model_name = st.selectbox("Выберите модель для визуализации", model_options, key="model_select")
            selected_model = next(r["Model_obj"] for r in results if r["Model"] == selected_model_name)
            st.session_state["best_model"] = selected_model

            # Пересчёт предсказаний: сначала модель предсказывает eps, затем по инверсии формулы получаем d
            try:
                eps_pred = selected_model.predict(X)
                df_show["eps_predicted"] = eps_pred
                df_show["d_predicted"] = df_show["eps_predicted"].apply(
                    lambda x: ((6.0 * surf_energy / density) ** (3/5) * 1e6 / (x ** (2/5)))
                )
                df_show["error_percent"] = df_show.apply(
                    lambda row: 100 * abs(row["d"] - row["d_predicted"]) / row["d"] if row["d"] != 0 else np.nan,
                    axis=1
                )
                r2_d = r2_score(df_show["d"], df_show["d_predicted"]) * 100
                mse_d = mean_squared_error(df_show["d"], df_show["d_predicted"])
                st.write(f"**Модель {selected_model_name}**: точность по d — R² = {r2_d:.2f}%, MSE = {mse_d:.4f}")
            except Exception as e:
                st.error(f"Ошибка при расчёте предсказаний: {e}")

            st.write("### Детальные предсказания:")
            st.dataframe(df_show)

            st.write("### Визуализация предсказаний")
            feature_cols = st.session_state["feature_cols"]

            # Если признак один – одномерная визуализация
            if len(feature_cols) == 1:
                feat = feature_cols[0]
                st.write(f"**Визуализация для одного признака: {feat}**")
                X_sorted = X.sort_values(by=feat)
                eps_pred_line = selected_model.predict(X_sorted)
                d_pred_line = np.array([
                    ((6.0 * surf_energy / density) ** (3/5) * 1e6 / (pred ** (2/5)))
                    for pred in eps_pred_line
                ])
                fig, ax = plt.subplots()
                ax.scatter(X[feat], df_show["d"], color="blue", label="Истинное d")
                ax.plot(X_sorted[feat], d_pred_line, color="red", label="Предсказанное d")
                ax.set_xlabel(feat)
                ax.set_ylabel("d")
                ax.set_title(f"Модель {selected_model_name}: d vs {feat}")
                ax.legend()
                st.pyplot(fig)
            # Если признаков два и более – 2D-визуализация
            elif len(feature_cols) >= 2:
                feat_x = st.selectbox("Признак X для визуализации", feature_cols, index=0, key="feat_x")
                remaining_feats = [f for f in feature_cols if f != feat_x]
                if remaining_feats:
                    feat_y = st.selectbox("Признак Y для визуализации", remaining_feats, index=0, key="feat_y")
                else:
                    feat_y = feat_x

                st.write(f"**Визуализация по признакам:** X = {feat_x}, Y = {feat_y}")
                x_range = np.linspace(X[feat_x].min(), X[feat_x].max(), 100)
                y_range = np.linspace(X[feat_y].min(), X[feat_y].max(), 100)
                xx, yy = np.meshgrid(x_range, y_range)

                # Создаём DataFrame для предсказаний на сетке
                grid_points = pd.DataFrame({feat_x: xx.ravel(), feat_y: yy.ravel()})
                # Для остальных признаков фиксируем их среднее значение
                for col in feature_cols:
                    if col not in [feat_x, feat_y]:
                        grid_points[col] = X[col].mean()
                # ВАЖНО! Приводим столбцы к тому же порядку, что и при обучении
                grid_points = grid_points[st.session_state["feature_cols"]]

                try:
                    eps_grid = selected_model.predict(grid_points)
                    d_grid = np.array([
                        ((6.0 * surf_energy / density) ** (3/5) * 1e6 / (pred ** (2/5)))
                        for pred in eps_grid
                    ])
                    d_grid = d_grid.reshape(xx.shape)

                    fig2, ax2 = plt.subplots()
                    contour = ax2.contourf(xx, yy, d_grid, cmap="viridis", alpha=0.7)
                    plt.colorbar(contour, ax=ax2, label="Предсказанное d")
                    ax2.scatter(X[feat_x], X[feat_y], c="red", edgecolors="k", label="Обучающие точки")
                    ax2.set_xlabel(feat_x)
                    ax2.set_ylabel(feat_y)
                    ax2.set_title(f"Модель {selected_model_name}: Предсказанное d по {feat_x} и {feat_y}")
                    ax2.legend()
                    st.pyplot(fig2)
                except Exception as e:
                    st.error(f"Ошибка при построении визуализации: {e}")

if __name__ == "__main__":
    show_menu()
