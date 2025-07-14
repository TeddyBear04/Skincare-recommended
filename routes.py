from flask import Blueprint, request, jsonify, render_template
from app.services.data_loader import skincare_df
from app.services.embedding import retrieve_top_products, search_multiple_ingredients
from app.services.llm import call_gemini_response
from app.services.prompt import build_prompt

main = Blueprint('main', __name__)

@main.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        skin_type = request.form.get("skin_type", "").strip()
        concerns_input = request.form.get("concerns", "").strip()
        ingredients_input = request.form.get("ingredients", "").strip()

        concerns = [c.strip().lower() for c in concerns_input.split(",") if c.strip()]
        ingredients = [i.strip().lower() for i in ingredients_input.split(",") if i.strip()]

        if not skin_type:
            return render_template("index.html", error="Please enter your skin type.")

        prompt = build_prompt(skin_type, concerns, skincare_df, ingredients=ingredients)
        ai_result = call_gemini_response(prompt)
        rag_result_df = retrieve_top_products(skin_type, concerns, skincare_df, ingredients)

        rag_result = []
        no_match_message = None
        if not rag_result_df.empty:
            rag_result = rag_result_df[["product_name", "concerns", "ingredient_list"]].to_dict(orient="records")
        else:
            no_match_message = "No products found matching your criteria."

        return render_template(
            "index.html",
            ai_result=ai_result,
            rag_result=rag_result,
            skin_type=skin_type,
            concerns=concerns_input,
            ingredients=ingredients_input,
            no_match_message=no_match_message
        )

    return render_template("index.html")


@main.route("/recommend", methods=["POST"])
def recommend_api():
    data = request.get_json()
    skin_type = data.get("skin_type", "").strip()
    concerns = [c.strip().lower() for c in data.get("concerns", []) if c.strip()]
    ingredients = [i.strip().lower() for i in data.get("ingredients", []) if i.strip()]

    prompt = build_prompt(skin_type, concerns, skincare_df, ingredients=ingredients)
    ai_result = call_gemini_response(prompt)
    grounded_products = retrieve_top_products(skin_type, concerns, skincare_df, ingredients)
    ingredient_results = search_multiple_ingredients(skincare_df, ingredients)

    grounded_result = []
    no_match_message = None
    if not grounded_products.empty:
        grounded_result = grounded_products[[
            'product_name', 'brand_name', 'concerns', 'ingredient_list', 'sim_score', 'total_score'
        ]].to_dict(orient="records")
    else:
        no_match_message = "No products found matching your criteria."

    return jsonify({
        "ai_result": ai_result,
        "grounded_result": grounded_result,
        "ingredient_results": ingredient_results,
        "no_match_message": no_match_message
    })
