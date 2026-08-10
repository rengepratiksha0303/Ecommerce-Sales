```python
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="E-Commerce Revenue Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce Revenue Prediction")
st.write(
    "Predict e-commerce revenue using KNN, ANN, CNN and RNN models."
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    file_path = "ecommerce_sales_analytics_5000.csv"

    df = pd.read_csv(file_path)

    return df


try:
    df = load_data()

except FileNotFoundError:
    st.error(
        "❌ ecommerce_sales_analytics_5000.csv was not found. "
        "Please place the CSV file in the same folder as app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"❌ Error while loading dataset: {e}")
    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
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

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "❌ Missing columns in dataset:\n"
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# DISPLAY DATASET
# ============================================================

with st.expander("📊 View Dataset"):

    st.write("Dataset Shape:", df.shape)

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# PREPROCESSING
# ============================================================

@st.cache_data
def preprocess_data(data):

    data = data.copy()

    # Convert date
    data["order_date"] = pd.to_datetime(
        data["order_date"],
        errors="coerce"
    )

    # Extract date features
    data["year"] = data["order_date"].dt.year
    data["month"] = data["order_date"].dt.month
    data["day"] = data["order_date"].dt.day
    data["day_of_week"] = data["order_date"].dt.dayofweek

    # Remove unnecessary columns
    data = data.drop(
        columns=[
            "order_id",
            "customer_id",
            "order_date"
        ],
        errors="ignore"
    )

    # Convert categorical columns
    categorical_columns = [
        "product_category",
        "region",
        "payment_method"
    ]

    for column in categorical_columns:

        if column in data.columns:

            data[column] = data[column].fillna("Unknown")

    data = pd.get_dummies(
        data,
        columns=categorical_columns,
        drop_first=True
    )

    # Convert all columns to numeric
    data = data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove missing values
    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.fillna(
        data.median(numeric_only=True)
    )

    # Target
    X = data.drop(
        columns=["revenue"]
    )

    y = data["revenue"]

    return X, y


try:

    X, y = preprocess_data(df)

except Exception as e:

    st.error(
        f"❌ Preprocessing error: {e}"
    )

    st.stop()


# ============================================================
# TRAIN TEST SPLIT
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
# TRAIN KNN
# ============================================================

@st.cache_resource
def train_knn(X_train, y_train):

    model = KNeighborsRegressor(
        n_neighbors=5,
        weights="distance"
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# TRAIN ANN
# ============================================================

@st.cache_resource
def train_ann(X_train, y_train):

    model = Sequential([
        Dense(
            128,
            activation="relu",
            input_shape=(X_train.shape[1],)
        ),

        Dropout(0.2),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.2),

        Dense(
            32,
            activation="relu"
        ),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    return model


# ============================================================
# TRAIN CNN
# ============================================================

@st.cache_resource
def train_cnn(X_train, y_train):

    # CNN requires 3D input
    X_train_cnn = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    model = Sequential([

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=(
                X_train.shape[1],
                1
            )
        ),

        MaxPooling1D(
            pool_size=2
        ),

        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        Flatten(),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.2),

        Dense(
            32,
            activation="relu"
        ),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    model.fit(
        X_train_cnn,
        y_train,
        validation_split=0.20,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    return model


# ============================================================
# TRAIN RNN
# ============================================================

@st.cache_resource
def train_rnn(X_train, y_train):

    # RNN requires 3D input
    X_train_rnn = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    model = Sequential([

        SimpleRNN(
            64,
            activation="tanh",
            return_sequences=True,
            input_shape=(
                X_train.shape[1],
                1
            )
        ),

        Dropout(0.2),

        SimpleRNN(
            32,
            activation="tanh"
        ),

        Dropout(0.2),

        Dense(
            32,
            activation="relu"
        ),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    model.fit(
        X_train_rnn,
        y_train,
        validation_split=0.20,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    return model


# ============================================================
# TRAIN ALL MODELS
# ============================================================

st.sidebar.header("⚙️ Model Training")

train_button = st.sidebar.button(
    "🚀 Train All Models"
)


if "models_trained" not in st.session_state:

    st.session_state.models_trained = False


if train_button:

    with st.spinner(
        "Training KNN, ANN, CNN and RNN models..."
    ):

        try:

            knn_model = train_knn(
                X_train_scaled,
                y_train
            )

            ann_model = train_ann(
                X_train_scaled,
                y_train
            )

            cnn_model = train_cnn(
                X_train_scaled,
                y_train
            )

            rnn_model = train_rnn(
                X_train_scaled,
                y_train
            )

            st.session_state.knn_model = knn_model
            st.session_state.ann_model = ann_model
            st.session_state.cnn_model = cnn_model
            st.session_state.rnn_model = rnn_model

            st.session_state.models_trained = True

            st.success(
                "✅ All four models trained successfully!"
            )

        except Exception as e:

            st.error(
                f"❌ Model training error: {e}"
            )


# ============================================================
# AUTOMATIC TRAINING
# ============================================================

if not st.session_state.models_trained:

    with st.spinner(
        "Training models for the first time..."
    ):

        try:

            knn_model = train_knn(
                X_train_scaled,
                y_train
            )

            ann_model = train_ann(
                X_train_scaled,
                y_train
            )

            cnn_model = train_cnn(
                X_train_scaled,
                y_train
            )

            rnn_model = train_rnn(
                X_train_scaled,
                y_train
            )

            st.session_state.knn_model = knn_model
            st.session_state.ann_model = ann_model
            st.session_state.cnn_model = cnn_model
            st.session_state.rnn_model = rnn_model

            st.session_state.models_trained = True

        except Exception as e:

            st.error(
                f"❌ Model training error: {e}"
            )

            st.stop()


# ============================================================
# MODEL EVALUATION
# ============================================================

st.header("📈 Model Performance")


knn_model = st.session_state.knn_model
ann_model = st.session_state.ann_model
cnn_model = st.session_state.cnn_model
rnn_model = st.session_state.rnn_model


# KNN prediction
knn_test_pred = knn_model.predict(
    X_test_scaled
)


# ANN prediction
ann_test_pred = ann_model.predict(
    X_test_scaled,
    verbose=0
).flatten()


# CNN prediction
X_test_cnn = X_test_scaled.reshape(
    X_test_scaled.shape[0],
    X_test_scaled.shape[1],
    1
)

cnn_test_pred = cnn_model.predict(
    X_test_cnn,
    verbose=0
).flatten()


# RNN prediction
X_test_rnn = X_test_scaled.reshape(
    X_test_scaled.shape[0],
    X_test_scaled.shape[1],
    1
)

rnn_test_pred = rnn_model.predict(
    X_test_rnn,
    verbose=0
).flatten()


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, mse, rmse, r2


knn_metrics = calculate_metrics(
    y_test,
    knn_test_pred
)

ann_metrics = calculate_metrics(
    y_test,
    ann_test_pred
)

cnn_metrics = calculate_metrics(
    y_test,
    cnn_test_pred
)

rnn_metrics = calculate_metrics(
    y_test,
    rnn_test_pred
)


performance_df = pd.DataFrame({

    "Model": [
        "KNN",
        "ANN",
        "CNN",
        "RNN"
    ],

    "MAE": [
        knn_metrics[0],
        ann_metrics[0],
        cnn_metrics[0],
        rnn_metrics[0]
    ],

    "MSE": [
        knn_metrics[1],
        ann_metrics[1],
        cnn_metrics[1],
        rnn_metrics[1]
    ],

    "RMSE": [
        knn_metrics[2],
        ann_metrics[2],
        cnn_metrics[2],
        rnn_metrics[2]
    ],

    "R2 Score": [
        knn_metrics[3],
        ann_metrics[3],
        cnn_metrics[3],
        rnn_metrics[3]
    ]
})


st.dataframe(
    performance_df.style.format({
        "MAE": "{:.2f}",
        "MSE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R2 Score": "{:.4f}"
    }),
    use_container_width=True
)


# ============================================================
# USER INPUT
# ============================================================

st.header("🔮 Revenue Prediction")


st.subheader("Enter Customer Order Details")


col1, col2, col3 = st.columns(3)


with col1:

    product_category = st.selectbox(
        "Product Category",
        sorted(
            df["product_category"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    region = st.selectbox(
        "Region",
        sorted(
            df["region"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        max_value=1000,
        value=1,
        step=1
    )


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
        max_value=100.0,
        value=0.0,
        step=0.5
    )

    payment_method = st.selectbox(
        "Payment Method",
        sorted(
            df["payment_method"]
            .dropna()
            .unique()
            .tolist()
        )
    )


with col3:

    delivery_days = st.number_input(
        "Delivery Days",
        min_value=0,
        max_value=100,
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
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Revenue",
    type="primary"
)


if predict_button:

    try:

        # ----------------------------------------
        # Create input dataframe
        # ----------------------------------------

        input_data = pd.DataFrame({
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


        # ----------------------------------------
        # One-hot encoding
        # ----------------------------------------

        input_data = pd.get_dummies(
            input_data,
            columns=[
                "product_category",
                "region",
                "payment_method"
            ],
            drop_first=True
        )


        # ----------------------------------------
        # Match training columns
        # ----------------------------------------

        input_data = input_data.reindex(
            columns=X.columns,
            fill_value=0
        )


        # ----------------------------------------
        # Convert to numeric
        # ----------------------------------------

        input_data = input_data.astype(float)


        # ----------------------------------------
        # Scale
        # ----------------------------------------

        input_scaled = scaler.transform(
            input_data
        )


        # ----------------------------------------
        # KNN prediction
        # ----------------------------------------

        knn_prediction = knn_model.predict(
            input_scaled
        )[0]


        # ----------------------------------------
        # ANN prediction
        # ----------------------------------------

        ann_prediction = ann_model.predict(
            input_scaled,
            verbose=0
        )[0][0]


        # ----------------------------------------
        # CNN prediction
        # ----------------------------------------

        input_cnn = input_scaled.reshape(
            1,
            input_scaled.shape[1],
            1
        )

        cnn_prediction = cnn_model.predict(
            input_cnn,
            verbose=0
        )[0][0]


        # ----------------------------------------
        # RNN prediction
        # ----------------------------------------

        input_rnn = input_scaled.reshape(
            1,
            input_scaled.shape[1],
            1
        )

        rnn_prediction = rnn_model.predict(
            input_rnn,
            verbose=0
        )[0][0]


        # ----------------------------------------
        # Display results
        # ----------------------------------------

        st.success(
            "✅ Revenue prediction completed!"
        )

        result_col1, result_col2, result_col3, result_col4 = st.columns(4)


        with result_col1:

            st.metric(
                "🤖 KNN",
                f"{knn_prediction:,.2f}"
            )


        with result_col2:

            st.metric(
                "🧠 ANN",
                f"{ann_prediction:,.2f}"
            )


        with result_col3:

            st.metric(
                "🖼️ CNN",
                f"{cnn_prediction:,.2f}"
            )


        with result_col4:

            st.metric(
                "🔄 RNN",
                f"{rnn_prediction:,.2f}"
            )


        # ----------------------------------------
        # Comparison table
        # ----------------------------------------

        prediction_df = pd.DataFrame({

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
            prediction_df.style.format({
                "Predicted Revenue": "{:,.2f}"
            }),
            use_container_width=True
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
    "E-Commerce Revenue Prediction | KNN • ANN • CNN • RNN"
)
```
