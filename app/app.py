from flask import Flask, render_template, request, session
from ai_assistant import ask_supply_chain_ai
from predictor import predict_delivery_risk, explain_prediction

import sqlite3
import pandas as pd
from pathlib import Path
import os


# ============================================================
# APPLICATION SETUP
# ============================================================

BASE_APP_DIR = Path(__file__).resolve().parent

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "database" / "supply_chain.db"

MODEL_RESULTS_PATH = BASE_DIR / "data" / "model_results.csv"


app = Flask(
    __name__,
    template_folder=str(BASE_APP_DIR / "templates")
)


# Secret key required for Flask session
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "ai-supply-chain-project-secret-key"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    connection = get_db_connection()


    # --------------------------------------------------------
    # TOTAL ORDERS
    # --------------------------------------------------------

    total_orders = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM orders
        """
    ).fetchone()["count"]


    # --------------------------------------------------------
    # LATE ORDERS
    # --------------------------------------------------------

    late_orders = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM orders
        WHERE late_delivery_risk = 1
        """
    ).fetchone()["count"]


    # --------------------------------------------------------
    # LATE DELIVERY RATE
    # --------------------------------------------------------

    late_rate = round(
        (late_orders / total_orders) * 100,
        2
    )


    # --------------------------------------------------------
    # AVERAGE SCHEDULED SHIPPING DAYS
    # --------------------------------------------------------

    avg_scheduled_shipping_days = connection.execute(
        """
        SELECT
            ROUND(
                AVG(scheduled_shipping_days),
                2
            ) AS avg_days
        FROM orders
        """
    ).fetchone()["avg_days"]


    # --------------------------------------------------------
    # MARKET PERFORMANCE
    # --------------------------------------------------------

    market_data = connection.execute(
        """
        SELECT
            market,
            COUNT(*) AS orders,
            SUM(late_delivery_risk) AS late_orders,
            ROUND(
                100.0 *
                SUM(late_delivery_risk) /
                COUNT(*),
                2
            ) AS late_rate
        FROM orders
        GROUP BY market
        ORDER BY late_rate DESC
        """
    ).fetchall()


    # --------------------------------------------------------
    # SHIPPING MODE PERFORMANCE
    # --------------------------------------------------------

    shipping_data = connection.execute(
        """
        SELECT
            shipping_mode,
            COUNT(*) AS orders,
            SUM(late_delivery_risk) AS late_orders,
            ROUND(
                100.0 *
                SUM(late_delivery_risk) /
                COUNT(*),
                2
            ) AS late_rate
        FROM orders
        GROUP BY shipping_mode
        ORDER BY late_rate DESC
        """
    ).fetchall()


    # --------------------------------------------------------
    # MONTHLY DELIVERY TREND
    # --------------------------------------------------------

    monthly_data = connection.execute(
        """
        SELECT
            substr(order_date, 1, 7) AS month,
            COUNT(*) AS orders,
            SUM(late_delivery_risk) AS late_orders,
            ROUND(
                100.0 *
                SUM(late_delivery_risk) /
                COUNT(*),
                2
            ) AS late_rate
        FROM orders
        GROUP BY month
        ORDER BY month
        """
    ).fetchall()


    connection.close()


    return render_template(
        "dashboard.html",

        total_orders=total_orders,

        late_orders=late_orders,

        late_rate=late_rate,

        avg_scheduled_shipping_days=
            avg_scheduled_shipping_days,

        market_data=market_data,

        shipping_data=shipping_data,

        monthly_data=monthly_data
    )

# ============================================================
# RISK PREDICTION PAGE
# ============================================================

@app.route("/risk-prediction")
def risk_prediction_page():

    return render_template(
        "index.html"
    )
# ============================================================
# RISK PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # COLLECT FORM DATA
        # ----------------------------------------------------

        order_data = {

            "order_type":
                request.form["order_type"],

            "market":
                request.form["market"],

            "order_city":
                request.form["order_city"],

            "order_country":
                request.form["order_country"],

            "order_region":
                request.form["order_region"],

            "order_state":
                request.form["order_state"],

            "shipping_mode":
                request.form["shipping_mode"],

            "scheduled_shipping_days":
                int(
                    request.form[
                        "scheduled_shipping_days"
                    ]
                ),

            "customer_segment":
                request.form["customer_segment"],

            "item_count":
                int(
                    request.form["item_count"]
                ),

            "distinct_product_count":
                int(
                    request.form[
                        "distinct_product_count"
                    ]
                ),

            "total_quantity":
                int(
                    request.form["total_quantity"]
                ),

            "avg_product_price":
                float(
                    request.form[
                        "avg_product_price"
                    ]
                ),

            "total_discount":
                float(
                    request.form["total_discount"]
                ),

            "avg_discount_rate":
                float(
                    request.form[
                        "avg_discount_rate"
                    ]
                ),

            "order_year":
                int(
                    request.form["order_year"]
                ),

            "order_month":
                int(
                    request.form["order_month"]
                ),

            "order_day_of_week":
                int(
                    request.form[
                        "order_day_of_week"
                    ]
                )
        }


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = predict_delivery_risk(
            order_data
        )


        # ----------------------------------------------------
        # SHAP EXPLANATION
        # ----------------------------------------------------

        explanation = explain_prediction(
            order_data,
            top_n=5
        )


        # ----------------------------------------------------
        # CONVERT SHAP RESULTS TO JSON-SAFE VALUES
        # ----------------------------------------------------

        positive_factors = explanation["positive"]

        negative_factors = explanation["negative"]


        # Convert pandas Series to dictionaries

        if hasattr(
            positive_factors,
            "to_dict"
        ):

            positive_factors = (
                positive_factors.to_dict()
            )


        if hasattr(
            negative_factors,
            "to_dict"
        ):

            negative_factors = (
                negative_factors.to_dict()
            )


        # Convert keys and values to
        # standard Python types

        positive_factors = {

            str(key):
                float(value)

            for key, value
            in positive_factors.items()
        }


        negative_factors = {

            str(key):
                float(value)

            for key, value
            in negative_factors.items()
        }


        # ----------------------------------------------------
        # FINAL RESULT FOR RISK PREDICTION PAGE
        # ----------------------------------------------------

        result = {

            "prediction":
                int(
                    prediction["prediction"]
                ),

            "probability":
                round(
                    float(
                        prediction["probability"]
                    ) * 100,
                    2
                ),

            "risk":
                prediction["risk"],

            "positive":
                positive_factors,

            "negative":
                negative_factors
        }


        # ----------------------------------------------------
        # SAVE LATEST PREDICTION FOR AI ASSISTANT
        # ----------------------------------------------------

        session["latest_prediction"] = {
            "prediction": int(result["prediction"]),
            "probability": float(result["probability"]),
            "risk": str(result["risk"]),
            "positive": {
                str(key): float(value)
                for key, value in positive_factors.items()
                },
                "negative": {
                    str(key): float(value)
                    for key, value in negative_factors.items()
                    },
                    "order_data": {
                        str(key): value
                        for key, value in order_data.items()
                        }
                        }


        # ----------------------------------------------------
        # SHOW RISK PREDICTION PAGE
        # ----------------------------------------------------

        return render_template(
            "index.html",
            result=result
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.route(
    "/model-performance",
    methods=["GET"]
)
def model_performance():

    model_results = []


    if MODEL_RESULTS_PATH.exists():

        model_df = pd.read_csv(
            MODEL_RESULTS_PATH
        )

        model_results = model_df.to_dict(
            orient="records"
        )


    return render_template(
        "model_performance.html",
        model_results=model_results
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

@app.route(
    "/business-insights",
    methods=["GET"]
)
def business_insights():

    connection = get_db_connection()


    # --------------------------------------------------------
    # TOTAL ORDERS
    # --------------------------------------------------------

    total_orders = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM orders
        """
    ).fetchone()["count"]


    # --------------------------------------------------------
    # LATE ORDERS
    # --------------------------------------------------------

    late_orders = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM orders
        WHERE late_delivery_risk = 1
        """
    ).fetchone()["count"]


    # --------------------------------------------------------
    # OVERALL LATE RATE
    # --------------------------------------------------------

    late_rate = round(
        (late_orders / total_orders) * 100,
        2
    )


    # --------------------------------------------------------
    # SHIPPING MODE ANALYSIS
    # --------------------------------------------------------

    shipping_data = connection.execute(
        """
        SELECT
            shipping_mode,
            COUNT(*) AS orders,
            SUM(late_delivery_risk) AS late_orders,
            ROUND(
                100.0 *
                SUM(late_delivery_risk) /
                COUNT(*),
                2
            ) AS late_rate
        FROM orders
        GROUP BY shipping_mode
        ORDER BY late_rate DESC
        """
    ).fetchall()


    # --------------------------------------------------------
    # MARKET ANALYSIS
    # --------------------------------------------------------

    market_data = connection.execute(
        """
        SELECT
            market,
            COUNT(*) AS orders,
            SUM(late_delivery_risk) AS late_orders,
            ROUND(
                100.0 *
                SUM(late_delivery_risk) /
                COUNT(*),
                2
            ) AS late_rate
        FROM orders
        GROUP BY market
        ORDER BY late_rate DESC
        """
    ).fetchall()


    # --------------------------------------------------------
    # MONTHLY ANALYSIS
    # --------------------------------------------------------

    monthly_data = connection.execute(
        """
        SELECT
            substr(order_date, 1, 7) AS month,
            COUNT(*) AS orders,
            SUM(late_delivery_risk) AS late_orders,
            ROUND(
                100.0 *
                SUM(late_delivery_risk) /
                COUNT(*),
                2
            ) AS late_rate
        FROM orders
        GROUP BY month
        ORDER BY month
        """
    ).fetchall()


    connection.close()


    # --------------------------------------------------------
    # HIGHEST / LOWEST SHIPPING MODE
    # --------------------------------------------------------

    highest_shipping = shipping_data[0]

    lowest_shipping = shipping_data[-1]


    # --------------------------------------------------------
    # HIGHEST / LOWEST MARKET
    # --------------------------------------------------------

    highest_market = market_data[0]

    lowest_market = market_data[-1]


    # --------------------------------------------------------
    # MODEL RESULTS
    # --------------------------------------------------------

    model_results = []


    if MODEL_RESULTS_PATH.exists():

        model_df = pd.read_csv(
            MODEL_RESULTS_PATH
        )

        model_results = model_df.to_dict(
            orient="records"
        )


    return render_template(
        "business_insights.html",

        total_orders=total_orders,

        late_orders=late_orders,

        late_rate=late_rate,

        shipping_data=shipping_data,

        market_data=market_data,

        monthly_data=monthly_data,

        highest_shipping=highest_shipping,

        lowest_shipping=lowest_shipping,

        highest_market=highest_market,

        lowest_market=lowest_market,

        model_results=model_results
    )


# ============================================================
# AI ASSISTANT
# ============================================================

@app.route(
    "/ai-assistant",
    methods=["GET", "POST"]
)
def ai_assistant():

    answer = None

    question = ""


    # --------------------------------------------------------
    # GET LATEST RISK PREDICTION
    # --------------------------------------------------------

    latest_prediction = session.get(
        "latest_prediction"
    )


    # --------------------------------------------------------
    # PROCESS AI QUESTION
    # --------------------------------------------------------

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()


        if question:

            try:

                answer = ask_supply_chain_ai(

                    question,

                    prediction_context=
                        latest_prediction
                )


            except Exception as e:

                answer = (
                    "Sorry, the AI assistant "
                    "could not process your question. "
                    f"Error: {str(e)}"
                )


    # --------------------------------------------------------
    # RENDER AI ASSISTANT
    # --------------------------------------------------------

    return render_template(

        "ai_assistant.html",

        question=question,

        answer=answer,

        latest_prediction=
            latest_prediction
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )