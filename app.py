```python
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv1D,
    MaxPooling1D,
    Flatten,
    SimpleRNN
)
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="E-Commerce Sales Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce Sales Prediction")
st.write(
    "KNN + ANN + CNN + RNN Revenue Prediction"
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "ecommerce_sales_analytics_5000.csv"

CATEGORICAL_COLUMNS = [
    "product_category",
    "region",
    "payment_method"
]

REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "product_category",
    "region",
    "quantity",
    "unit_price",
    "discount",
    "payment_method",
    "delivery_days",
    "customer_rating",
    "revenue"
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(DATA_FILE)

    return df


try:

    df = load_dataset()

except FileNotFoundError:

    st.error(
        f"❌ File '{DATA_FILE}' was not found."
    )

    st.info(
        "Put the CSV file in the same GitHub repository as app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading CSV: {e}"
    )

    st.stop()


# ============================================================
# CHECK COLUMNS
# ============================================================

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:

    st.error(
        "❌ Missing columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander("📊 Dataset Information"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Target",
            "Revenue"
        )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# PREPROCESS DATA
# ============================================================

@st.cache_data
def prepare_data(data):

    data = data.copy()

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    data["order_date"] = pd.to_datetime(
        data["order_date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Extract date features
    # --------------------------------------------------------

    data["year"] = data["order_date"].dt.year
    data["month"] = data["order_date"].dt.month
    data["day"] = data["order_date"].dt.day
    data["day_of_week"] = data["order_date"].dt.dayofweek

    # --------------------------------------------------------
    # Remove ID/date columns
    # --------------------------------------------------------

    data.drop(
        columns=[
            "order_id",
            "customer_id",
            "order_date"
        ],
        inplace=True,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Fill categorical missing values
    # --------------------------------------------------------

    for col in CATEGORICAL_COLUMNS:

        if col in data.columns:

            data[col] = data[col].fillna(
                "Unknown"
            )

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    data = pd.get_dummies(
        data,
        columns=CATEGORICAL_COLUMNS,
        drop_first=False
    )

    # --------------------------------------------------------
    # Convert everything to numeric
    # --------------------------------------------------------

    data = data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # --------------------------------------------------------
    # Fill numerical missing values
    # --------------------------------------------------------

    data.fillna(
        data.median(numeric_only=True),
        inplace=True
    )

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    data.fillna(
        0,
        inplace=True
    )

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X = data.drop(
        columns=["revenue"]
    )

    y = data["revenue"].astype(float)

    return X, y


try:

    X, y = prepare_data(df)

except Exception as e:

    st.error(
        f"❌ Data preprocessing error: {e}"
    )

    st.stop()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# KNN
# ============================================================

@st.cache_resource
def create_knn(
    X_train_data,
    y_train_data
):

    model = KNeighborsRegressor(
        n_neighbors=5,
        weights="distance"
    )

    model.fit(
        X_train_data,
        y_train_data
    )

    return model


# ============================================================
# ANN
# ============================================================

@st.cache_resource
def create_ann(
    number_of_features,
    X_train_data,
    y_train_data
):

    model = Sequential([

        Dense(
            128,
            activation="relu",
            input_shape=(number_of_features,)
        ),

        Dropout(0.20),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.20),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            1,
            activation="linear"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train_data,
        y_train_data,
        validation_split=0.20,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    return model


# ============================================================
# CNN
# ============================================================

@st.cache_resource
def create_cnn(
    number_of_features,
    X_train_data,
    y_train_data
):

    X_cnn = X_train_data.reshape(
        X_train_data.shape[0],
        X_train_data.shape[1],
        1
    )

    model = Sequential([

        Conv1D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
            input_shape=(
                number_of_features,
                1
            )
        ),

        MaxPooling1D(
            pool_size=2
        ),

        Conv1D(
            filters=16,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        Flatten(),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.20),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            1,
            activation="linear"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_cnn,
        y_train_data,
        validation_split=0.20,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    return model


# ============================================================
# RNN
# ============================================================

@st.cache_resource
def create_rnn(
    number_of_features,
    X_train_data,
    y_train_data
):

    X_rnn = X_train_data.reshape(
        X_train_data.shape[0],
        X_train_data.shape[1],
        1
    )

    model = Sequential([

        SimpleRNN(
            32,
            activation="tanh",
            return_sequences=True,
            input_shape=(
                number_of_features,
                1
            )
        ),

        Dropout(0.20),

        SimpleRNN(
            16,
            activation="tanh"
        ),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            1,
            activation="linear"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_rnn,
        y_train_data,
        validation_split=0.20,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    return model


# ============================================================
# TRAIN MODELS
# ============================================================

if "models_ready" not in st.session_state:

    st.session_state.models_ready = False


if not st.session_state.models_ready:

    progress = st.progress(0)

    status = st.empty()

    try:

        status.write("🔵 Training KNN...")
        knn_model = create_knn(
            X_train_scaled,
            y_train
        )

        progress.progress(25)

        status.write("🟢 Training ANN...")
        ann_model = create_ann(
            X_train_scaled.shape[1],
            X_train_scaled,
            y_train
        )

        progress.progress(50)

        status.write("🟠 Training CNN...")
        cnn_model = create_cnn(
            X_train_scaled.shape[1],
            X_train_scaled,
            y_train
        )

        progress.progress(75)

        status.write("🟣 Training RNN...")
        rnn_model = create_rnn(
            X_train_scaled.shape[1],
            X_train_scaled,
            y_train
        )

        progress.progress(100)

        st.session_state.knn_model = knn_model
        st.session_state.ann_model = ann_model
        st.session_state.cnn_model = cnn_model
        st.session_state.rnn_model = rnn_model

        st.session_state.models_ready = True

        status.success(
            "✅ All models trained successfully!"
        )

    except Exception as e:

        st.error(
            f"❌ Model training failed: {e}"
        )

        st.stop()


# ============================================================
# LOAD MODELS FROM SESSION
# ============================================================

knn_model = st.session_state.knn_model
ann_model = st.session_state.ann_model
cnn_model = st.session_state.cnn_model
rnn_model = st.session_state.rnn_model


# ============================================================
# MODEL EVALUATION
# ============================================================

@st.cache_data
def evaluate_models(
    X_test_data,
    y_test_data
):

    # KNN
    knn_pred = knn_model.predict(
        X_test_data
    )

    # ANN
    ann_pred = ann_model.predict(
        X_test_data,
        verbose=0
    ).flatten()

    # CNN
    X_cnn_test = X_test_data.reshape(
        X_test_data.shape[0],
        X_test_data.shape[1],
        1
    )

    cnn_pred = cnn_model.predict(
        X_cnn_test,
        verbose=0
    ).flatten()

    # RNN
    X_rnn_test = X_test_data.reshape(
        X_test_data.shape[0],
        X_test_data.shape[1],
        1
    )

    rnn_pred = rnn_model.predict(
        X_rnn_test,
        verbose=0
    ).flatten()

    predictions = {
        "KNN": knn_pred,
        "ANN": ann_pred,
        "CNN": cnn_pred,
        "RNN": rnn_pred
    }

    rows = []

    for model_name, prediction in predictions.items():

        mae = mean_absolute_error(
            y_test_data,
            prediction
        )

        mse = mean_squared_error(
            y_test_data,
            prediction
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test_data,
            prediction
        )

        rows.append({
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        })

    metrics_df = pd.DataFrame(rows)

    return metrics_df


# ============================================================
# PERFORMANCE
# ============================================================

st.header("📈 Model Performance")

metrics_df = evaluate_models(
    X_test_scaled,
    y_test
)

st.dataframe(
    metrics_df.style.format({
        "MAE": "{:,.2f}",
        "MSE": "{:,.2f}",
        "RMSE": "{:,.2f}",
        "R2 Score": "{:.4f}"
    }),
    use_container_width=True
)


# ============================================================
# USER INPUT
# ============================================================

st.header("🔮 Predict Revenue")

st.write(
    "Enter the order details below."
)


col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    product_categories = sorted(
        df["product_category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    regions = sorted(
        df["region"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    product_category = st.selectbox(
        "Product Category",
        product_categories
    )

    region = st.selectbox(
        "Region",
        regions
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=1,
        step=1
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    unit_price = st.number_input(
        "Unit Price",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    discount = st.number_input(
        "Discount",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

    payment_methods = sorted(
        df["payment_method"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    payment_method = st.selectbox(
        "Payment Method",
        payment_methods
    )


# ------------------------------------------------------------
# COLUMN 3
# ------------------------------------------------------------

with col3:

    delivery_days = st.number_input(
        "Delivery Days",
        min_value=0,
        value=3,
        step=1
    )

    customer_rating = st.number_input(
        "Customer Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

    order_date = st.date_input(
        "Order Date"
    )


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Revenue",
    type="primary"
):

    try:

        # ----------------------------------------------------
        # Create raw input
        # ----------------------------------------------------

        input_df = pd.DataFrame({

            "product_category": [
                product_category
            ],

            "region": [
                region
            ],

            "quantity": [
                quantity
            ],

            "unit_price": [
                unit_price
            ],

            "discount": [
                discount
            ],

            "payment_method": [
                payment_method
            ],

            "delivery_days": [
                delivery_days
            ],

            "customer_rating": [
                customer_rating
            ],

            "year": [
                order_date.year
            ],

            "month": [
                order_date.month
            ],

            "day": [
                order_date.day
            ],

            "day_of_week": [
                order_date.weekday()
            ]
        })


        # ----------------------------------------------------
        # One-hot encode
        # ----------------------------------------------------

        input_df = pd.get_dummies(
            input_df,
            columns=CATEGORICAL_COLUMNS,
            drop_first=False
        )


        # ----------------------------------------------------
        # Match exact training columns
        # ----------------------------------------------------

        input_df = input_df.reindex(
            columns=X.columns,
            fill_value=0
        )


        # ----------------------------------------------------
        # Convert to float
        # ----------------------------------------------------

        input_df = input_df.astype(float)


        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        input_scaled = scaler.transform(
            input_df
        )


        # ----------------------------------------------------
        # KNN prediction
        # ----------------------------------------------------

        knn_prediction = float(
            knn_model.predict(
                input_scaled
            )[0]
        )


        # ----------------------------------------------------
        # ANN prediction
        # ----------------------------------------------------

        ann_prediction = float(
            ann_model.predict(
                input_scaled,
                verbose=0
            )[0][0]
        )


        # ----------------------------------------------------
        # CNN prediction
        # ----------------------------------------------------

        input_cnn = input_scaled.reshape(
            1,
            input_scaled.shape[1],
            1
        )

        cnn_prediction = float(
            cnn_model.predict(
                input_cnn,
                verbose=0
            )[0][0]
        )


        # ----------------------------------------------------
        # RNN prediction
        # ----------------------------------------------------

        input_rnn = input_scaled.reshape(
            1,
            input_scaled.shape[1],
            1
        )

        rnn_prediction = float(
            rnn_model.predict(
                input_rnn,
                verbose=0
            )[0][0]
        )


        # ----------------------------------------------------
        # Prevent negative revenue
        # ----------------------------------------------------

        knn_prediction = max(
            0,
            knn_prediction
        )

        ann_prediction = max(
            0,
            ann_prediction
        )

        cnn_prediction = max(
            0,
            cnn_prediction
        )

        rnn_prediction = max(
            0,
            rnn_prediction
        )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.success(
            "✅ Revenue prediction completed!"
        )

        st.subheader(
            "💰 Predicted Revenue"
        )

        result1, result2, result3, result4 = st.columns(4)


        with result1:

            st.metric(
                "🤖 KNN",
                f"{knn_prediction:,.2f}"
            )


        with result2:

            st.metric(
                "🧠 ANN",
                f"{ann_prediction:,.2f}"
            )


        with result3:

            st.metric(
                "🖼️ CNN",
                f"{cnn_prediction:,.2f}"
            )


        with result4:

            st.metric(
                "🔄 RNN",
                f"{rnn_prediction:,.2f}"
            )


        # ====================================================
        # COMPARISON
        # ====================================================

        comparison_df = pd.DataFrame({

            "Model": [
                "KNN",
                "ANN",
                "CNN",
                "RNN"
            ],

            "Predicted Revenue": [
                knn_prediction,
                ann_prediction,
                cnn_prediction,
                rnn_prediction
            ]
        })


        st.subheader(
            "📊 Prediction Comparison"
        )

        st.dataframe(
            comparison_df.style.format({
                "Predicted Revenue": "{:,.2f}"
            }),
            use_container_width=True
        )


        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        best_model_row = metrics_df.loc[
            metrics_df["R2 Score"].idxmax()
        ]

        st.info(
            f"🏆 Best performing model: "
            f"**{best_model_row['Model']}** "
            f"with R² Score = "
            f"**{best_model_row['R2 Score']:.4f}**"
        )


    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "E-Commerce Sales Analytics | "
    "KNN • ANN • CNN • RNN"
)
```
