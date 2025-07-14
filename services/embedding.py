import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def retrieve_top_products(skin_type, concerns, df, ingredients=None, top_k=3, model=None):
    from sentence_transformers import SentenceTransformer
    model = model or SentenceTransformer('all-MiniLM-L6-v2')

    query_text = f"{skin_type} skin with concerns: {', '.join(concerns)}"
    query_vec = model.encode(query_text)
    sims = cosine_similarity([query_vec], list(df['embedding']))
    df = df.copy()
    df['sim_score'] = sims[0]

    # Top 100 trước khi lọc theo ingredient
    top_df = df.sort_values(by='sim_score', ascending=False).head(100)

    if ingredients:
        ingredients = [i.lower().strip() for i in ingredients]

        def match_ingredients(ing_list):
            return any(any(target_ing in ing for ing in ing_list) for target_ing in ingredients)

        filtered_df = top_df[top_df['ingredient_list'].apply(match_ingredients)]

        if not filtered_df.empty:
            top_df = filtered_df
        else:
            print("⚠️ No product matches the exact ingredients. Showing top relevant ones.")

    top_df = top_df.dropna(subset=['sentiment_score_norm', 'rating_norm'])

    if top_df.empty:
        return pd.DataFrame()

    top_df['total_score'] = (
        0.4 * top_df['sim_score'] +
        0.3 * top_df['sentiment_score_norm'] +
        0.2 * top_df['rating_norm'] +
        0.1 * top_df['recommend_norm']
    )

    return top_df.sort_values(by='total_score', ascending=False).head(top_k)


def search_by_ingredient(df, ingredient_name, top_k=3):
    ingredient_name = ingredient_name.lower()
    matched_df = df[
        df['ingredient_list'].apply(lambda ings: any(ingredient_name in ing for ing in ings))
    ]
    return matched_df.head(top_k)[[
        'product_name', 'brand_name', 'skin_type', 'concerns', 'ingredients', 'price_usd'
    ]]

def search_multiple_ingredients(df, ingredients, top_k=3):
    results = {}
    for ing in ingredients:
        results[ing] = search_by_ingredient(df, ing, top_k).to_dict(orient='records')
    return results

def filter_by_rating(products_df, min_rating=0):
    return products_df[products_df['rating'] >= min_rating]
