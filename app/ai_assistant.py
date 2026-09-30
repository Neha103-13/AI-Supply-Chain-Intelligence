from pathlib import Path
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import os
import re


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "supply_chain.db"

MODEL_RESULTS_PATH = BASE_DIR / "data" / "model_results.csv"


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ============================================================
# OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# RETRIEVE SUPPLY CHAIN DATA
# ============================================================

def get_supply_chain_context():

    connection = get_db_connection()

    # --------------------------------------------------------
    # OVERALL DELIVERY PERFORMANCE
    # --------------------------------------------------------

    overall = connection.execute(
        """
        SELECT
            COUNT(*) AS total_orders,
            SUM(late_delivery_risk) AS late_orders,
            ROUND(
                100.0 * SUM(late_delivery_risk) / COUNT(*),
                2
            ) AS late_rate
        FROM orders
        """
    ).fetchone()

    # --------------------------------------------------------
    # SHIPPING MODE PERFORMANCE
    # --------------------------------------------------------

    shipping = connection.execute(
        """
        SELECT
            shipping_mode,
            COUNT(*) AS orders,
            SUM(late_delivery_risk) AS late_orders,
            ROUND(
                100.0 * SUM(late_delivery_risk) / COUNT(*),
                2
            ) AS late_rate
        FROM orders
        GROUP BY shipping_mode
        ORDER BY late_rate DESC
        """
    ).fetchall()

    # --------------------------------------------------------
    # MARKET PERFORMANCE
    # --------------------------------------------------------

    market = connection.execute(
        """
        SELECT
            market,
            COUNT(*) AS orders,
            SUM(late_delivery_risk) AS late_orders,
            ROUND(
                100.0 * SUM(late_delivery_risk) / COUNT(*),
                2
            ) AS late_rate
        FROM orders
        GROUP BY market
        ORDER BY late_rate DESC
        """
    ).fetchall()

    # --------------------------------------------------------
    # MONTHLY DELIVERY TREND
    # --------------------------------------------------------

    monthly = connection.execute(
        """
        SELECT
            substr(order_date, 1, 7) AS month,
            COUNT(*) AS orders,
            SUM(late_delivery_risk) AS late_orders,
            ROUND(
                100.0 * SUM(late_delivery_risk) / COUNT(*),
                2
            ) AS late_rate
        FROM orders
        GROUP BY month
        ORDER BY month
        """
    ).fetchall()

    connection.close()

    # Convert SQLite rows to dictionaries

    shipping = [dict(row) for row in shipping]

    market = [dict(row) for row in market]

    monthly = [dict(row) for row in monthly]

    # --------------------------------------------------------
    # MACHINE LEARNING RESULTS
    # --------------------------------------------------------

    model_results = []

    if MODEL_RESULTS_PATH.exists():

        model_df = pd.read_csv(MODEL_RESULTS_PATH)

        model_results = model_df.to_dict(
            orient="records"
        )

    # --------------------------------------------------------
    # COMPLETE PROJECT CONTEXT
    # --------------------------------------------------------

    context = {

        "overall": dict(overall),

        "shipping_modes": shipping,

        "markets": market,

        "monthly_trends": monthly,

        "model_results": model_results
    }

    return context


# ============================================================
# CLEAN AI RESPONSE
# ============================================================

def clean_ai_response(answer):

    if not answer:
        return ""

    # Remove Markdown bold
    answer = answer.replace("**", "")

    # Remove Markdown italic markers
    answer = answer.replace("__", "")
    answer = answer.replace("*", "")

    # Remove Markdown headings
    answer = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        answer,
        flags=re.MULTILINE
    )

    # Remove Markdown code fences
    answer = answer.replace("```", "")

    # Remove excessive blank lines
    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer
    )

    # Remove unnecessary spaces at line beginnings
    answer = re.sub(
        r"\n[ \t]+",
        "\n",
        answer
    )

    return answer.strip()


# ============================================================
# ASK SUPPLY CHAIN AI
# ============================================================

def ask_supply_chain_ai(
    question,
    prediction_context=None
):

    context = get_supply_chain_context()
    if prediction_context:
        context["latest_prediction"] = prediction_context
    else:
        context["latest_prediction"] = (
            "No individual risk prediction is currently available."
        )

    prompt = f"""



You are an AI Supply Chain Intelligence Assistant.

You help users understand a supply-chain analytics project
using SQL analytics, machine learning and explainable AI.

The actual project information retrieved from the database
and machine-learning results is provided below.

============================================================
PROJECT DATA
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
ANSWERING RULES
============================================================

1. Answer the user's question clearly and directly.

2. Use ONLY the project data provided above.

3. Never invent statistics, values or model results.

4. When giving numerical information, use the actual values
   from the project data.

5. Explain machine-learning concepts in simple business
   language.

6. Distinguish observed patterns from causal relationships.
   Do not claim that one factor causes another unless the
   project data establishes causality.

7. When discussing model performance, correctly distinguish
   Accuracy, Precision, Recall, F1 and ROC-AUC.

8. If the requested information is not available in the
   project data, say:
   This information is not available in the current project data.

9. Keep answers concise but useful.

10. Use simple bullet points when useful.

11. DO NOT use Markdown formatting.

12. DO NOT use asterisks.

13. DO NOT use Markdown bold formatting.

14. DO NOT use hashtags for headings.

15. DO NOT use code blocks.

16. Return clean plain text suitable for displaying directly
   inside a web application.

17. Do not begin the response with unnecessary phrases such as
    "Sure!" or "Certainly!" unless appropriate.

18. Focus directly on answering the user's question.

19. When explaining an individual risk prediction, use the
    latest_prediction information provided in the project context.

20. Convert technical feature names into human-readable business
    names. Use these mappings:

    shipping_mode → Shipping Mode
    order_type → Order Type
    scheduled_shipping_days → Scheduled Shipping Days
    order_city → Order City
    order_state → Order State
    order_country → Order Country
    order_region → Order Region
    market → Market
    customer_segment → Customer Segment
    item_count → Number of Items
    distinct_product_count → Number of Different Products
    total_quantity → Total Quantity
    avg_product_price → Average Product Price
    total_discount → Total Discount
    avg_discount_rate → Average Discount Rate
    order_year → Order Year
    order_month → Order Month
    order_day_of_week → Order Day of Week

21. When presenting SHAP contributions, do not expose Python
    variable names such as "shipping_mode" or "avg_product_price".

22. Explain SHAP values in business-friendly language.
    For example:

    "Shipping Mode made a strong positive contribution to the
    model's high-risk prediction."

23. Clearly distinguish model contribution from causation.
    Do not say that a SHAP factor caused the delivery delay.

24. If the SHAP contribution is positive, explain that it pushed
    the model toward a higher late-delivery risk.

25. If the SHAP contribution is negative, explain that it pushed
    the model toward a lower late-delivery risk.

26. Include the numerical SHAP value when useful, but do not
    describe it as a percentage or probability.

27. Keep the explanation concise, professional, and suitable
    for a business intelligence dashboard.

28. Do not invent factors, SHAP values, probabilities, or
    prediction details that are not present in the supplied
    project context.
"""

    # --------------------------------------------------------
    # OPENROUTER FREE MODEL
    # --------------------------------------------------------

    response = client.chat.completions.create(

        model="openrouter/free",

        messages=[

            {
                "role": "system",
                "content": (
                    "You are a reliable Supply Chain "
                    "Intelligence Assistant. "
                    "Use only the supplied project data. "
                    "Return clean plain text without Markdown."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

    )

    answer = response.choices[0].message.content

    # Clean any Markdown that the model may still return

    answer = clean_ai_response(answer)

    return answer


# ============================================================
# TEST THE AI ASSISTANT
# ============================================================

if __name__ == "__main__":

    question = (
        "Which shipping mode has the highest "
        "late delivery rate?"
    )

    answer = ask_supply_chain_ai(question)

    print()
    print("=" * 55)
    print("AI SUPPLY CHAIN ASSISTANT")
    print("=" * 55)
    print()

    print(answer)

    print()
    print("=" * 55)