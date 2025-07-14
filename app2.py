import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import random
import google.generativeai as genai
import json
import os
import re

product_df = pd.read_csv('G:\language\Project2\TMG\product_info\product_info.csv')
review_paths = [
    # 'G:\language\Project2\TMG\Reviews\Reviews_0-250.csv',
    # 'G:\language\Project2\TMG\Reviews\Reviews_250-500.csv',
    # 'G:\language\Project2\TMG\Reviews\Reviews_500-750.csv',
    # 'G:\language\Project2\TMG\Reviews\Reviews_750-1250.csv',
    'G:\language\Project2\TMG\Reviews\Reviews_1250-end.csv'
]
review_df = pd.concat([pd.read_csv(p) for p in review_paths], ignore_index=True)
skincare_df = review_df.merge(product_df, on='product_id', how='inner')



for col in skincare_df.columns:
    if col.endswith('_x'):
        base = col[:-2]
        col_y = base + '_y'
        if col_y in skincare_df.columns:
            skincare_df[base] = skincare_df[col].combine_first(skincare_df[col_y])
            skincare_df.drop(columns=[col, col_y], inplace=True)

columns_to_keep = [
    'product_id',
    'product_name',
    'brand_name',
    'ingredients',
    'price_usd',
    'primary_category',
    'skin_type',
    'highlights',
    'review_title',
    'is_recommended',
    'rating'
]

skincare_df = skincare_df[columns_to_keep]

skincare_df = skincare_df[skincare_df['primary_category'].str.lower().str.contains('skincare', na=False)]

columns_to_check = [

    'product_id',
    'product_name',
    'brand_name',
    'ingredients',
    'price_usd',
    'primary_category',
    'skin_type',
    'highlights',
    'review_title',
    'is_recommended',
    'rating'

]
skincare_df = skincare_df.dropna(subset=columns_to_check)

import nltk
nltk.download('vader_lexicon')

# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import re

# #  Khởi tạo VADER
# sia = SentimentIntensityAnalyzer()

# #  Tạo danh sách stopwords tùy chỉnh
# custom_stopwords = set(STOPWORDS)
# custom_stopwords.update(['skin', 'product', 'use', 'face', 'really', 'make', 'buy', 'cream'])

# #  Làm sạch nội dung đánh giá văn bản
# def clean_text(text):
#     if isinstance(text, str):
#         text = text.lower()
#         text = re.sub(r'[^a-zA-Z\s]', '', text)
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text
#     return ''

# #  Tạo cột 'cleaned_review_text' dựa trên cột 'text'
# skincare_df['review_title'] = skincare_df['review_title'].apply(clean_text)

# #  Hàm trích từ theo cảm xúc (với tqdm để theo dõi tiến trình)
# def extract_vader_sentiment_words(text_list, sentiment='positive'):
#     result = []
#     for text in tqdm(text_list, desc=f"Đang xử lý từ {sentiment}"):
#         if not isinstance(text, str):
#             continue
#         words = text.split()
#         for word in words:
#             score = sia.polarity_scores(word)['compound']
#             if sentiment == 'positive' and score > 0.05:
#                 result.append(word.lower())
#             elif sentiment == 'negative' and score < -0.05:
#                 result.append(word.lower())
#     return result

# #  Trích từ trong review tích cực
# positive_reviews = skincare_df[skincare_df['is_recommended'] == 1]['review_title'].dropna().tolist()
# positive_words = extract_vader_sentiment_words(positive_reviews, sentiment='positive')
# all_positive_text = ' '.join(positive_words)

# #  Trích từ trong review tiêu cực
# negative_reviews = skincare_df[skincare_df['is_recommended'] == 0]['review_title'].dropna().tolist()
# negative_words = extract_vader_sentiment_words(negative_reviews, sentiment='negative')
# all_negative_text = ' '.join(negative_words)

# #  Tạo WordCloud cho từ tích cực
# if all_positive_text.strip():
#     plt.figure(figsize=(10, 5))
#     wc_pos = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords).generate(all_positive_text)
#     plt.imshow(wc_pos, interpolation='bilinear')
#     plt.title(' Positive Words in Positive Reviews (VADER + tqdm)')
#     plt.axis('off')
#     plt.savefig('vader_positive_words_tqdm.png')
#     plt.show()

# #  Tạo WordCloud cho từ tiêu cực
# if all_negative_text.strip():
#     plt.figure(figsize=(10, 5))
#     wc_neg = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords).generate(all_negative_text)
#     plt.imshow(wc_neg, interpolation='bilinear')
#     plt.title(' Negative Words in Negative Reviews (VADER + tqdm)')
#     plt.axis('off')
#     plt.savefig('vader_negative_words_tqdm.png')
#     plt.show()


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Khởi tạo VADER
sia = SentimentIntensityAnalyzer()

# 3. Phân tích sentiment từng review_title
skincare_df['review_sentiment'] = skincare_df['review_title'].apply(
    lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0
)


#  4. Tổng hợp điểm sentiment theo sản phẩm
product_sentiment_df = skincare_df.groupby('product_id').agg({
    'review_sentiment': 'mean',
    'rating': 'mean',
    'is_recommended': 'mean',
    'product_name': 'first',
    'brand_name': 'first',
    'skin_type': 'first'
}).rename(columns={
    'review_sentiment': 'avg_sentiment',
    'rating': 'avg_rating',
    'is_recommended': 'recommend_ratio'
})

#5. Chuẩn hóa các cột điểm
scaler = MinMaxScaler()
product_sentiment_df[['sentiment_score_norm', 'rating_norm', 'recommend_norm']] = scaler.fit_transform(
    product_sentiment_df[['avg_sentiment', 'avg_rating', 'recommend_ratio']]
)

#  6. Gộp lại vào skincare_df theo product_id
skincare_df = skincare_df.merge(
    product_sentiment_df[['sentiment_score_norm', 'rating_norm', 'recommend_norm']],
    on='product_id', how='left'
)

# ====== 1. Từ điển ánh xạ thành phần ➜ concern ======
def load_ingredient_concern_mapping(file_path):
    ingredient_to_concern = {}
    current_concern = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("=== ") and line.endswith(" ==="):
                current_concern = line.replace("===", "").strip().lower()
            elif line.startswith("- ") and current_concern:
                ingredient = line[2:].strip().lower()
                ingredient_to_concern[ingredient] = current_concern
    return ingredient_to_concern

# Load từ file mapping đã tạo
ingredient_to_concern = load_ingredient_concern_mapping("ingredient_to_concern_mapping.txt")

def clean_ingredient_text(raw_text):
    if not isinstance(raw_text, str):
        return []
    text = raw_text.lower()
    text = re.sub(r'\(.*?\)', '', text)  # bỏ ngoặc
    text = re.sub(r'\s+', ' ', text)  # chuẩn hóa khoảng trắng
    items = [re.sub(r'[^a-z0-9 ]', '', i.strip()) for i in text.split(',') if i.strip()]
    return items

def map_ingredients_to_concerns(ingredient_list):
    matched = set()
    for ing in ingredient_list:
        ing_norm = ing.strip().lower()
        for known_ing in ingredient_to_concern:
            if known_ing in ing_norm:
                matched.add(ingredient_to_concern[known_ing])
    return list(matched) if matched else ['unknown']

# ====== 4. Hàm chính xử lý DataFrame ======
def process_ingredients_column(skincare_df, model):
    # 1. Làm sạch ingredients và ánh xạ concern
    skincare_df['ingredient_list'] = skincare_df['ingredients'].apply(clean_ingredient_text)
    skincare_df['concerns'] = skincare_df['ingredient_list'].apply(map_ingredients_to_concerns)

    # 2. Encode theo batch
    concern_texts = skincare_df['concerns'].astype(str).tolist()
    embeddings = model.encode(concern_texts, batch_size=32, show_progress_bar=True)
    skincare_df['embedding'] = embeddings.tolist()

    return skincare_df


def search_by_ingredient(skincare_df, ingredient_name, top_k=3):
    ingredient_name = ingredient_name.lower()
    matched_df = skincare_df[
        skincare_df['ingredient_list'].apply(lambda ings: any(ingredient_name in ing for ing in ings))
    ]
    return matched_df.head(top_k)[[
        'product_name', 'brand_name', 'skin_type', 'concerns', 'ingredients', 'price_usd'
    ]]

def search_multiple_ingredients(skincare_df, ingredients, top_k=3):
    results = {}
    for ing in ingredients:
        results[ing] = search_by_ingredient(skincare_df, ing, top_k=top_k).to_dict(orient='records')
    return results

def build_prompt(skin_type, concerns, product_df):
    product_examples = product_df.sample(3, random_state=42).to_dict(orient='records')
    examples_str = "\n".join([
        f"{i+1}. {{\"product_name\": \"{p['product_name']}\", \"brand_name\": \"{p['brand_name']}\", \"skin_type\": \"{p['skin_type']}\", \"concerns\": \"{p['concerns']}\", \"ingredients\": \"{p['ingredients']}\", \"price\": \"{p['price_usd']}\"}}"
        for i, p in enumerate(product_examples)
    ])

    prompt = f"""
You are a skincare expert. Recommend 3 suitable products in JSON format for someone with {skin_type} skin and concerns: {', '.join(concerns)}.

Here are example products:
{examples_str}

 Do NOT use Markdown formatting, bullet points, or natural language explanations.
 Respond ONLY with a JSON **array** of 3 objects, and each object must contain:

- product_name (string)
- brand_name (string)
- skin_type (string)
- concerns (string or list)
- ingredients (string)
- price (e.g., "29.99 USD")
- justification (string: short explanation why this product matches)

Your response **must be valid JSON**, no extra text before or after.
"""
    return prompt

genai.configure(api_key="AIzaSyC2CE4fX4Og7IxpHgIvjX3WbnXc_53kQas")
models = list(genai.list_models())

for model in models:
    print(model.name)


def clean_and_parse_json(response_text):
    cleaned = re.sub(r"```(?:json)?", "", response_text).strip("`\n ")
    try:
        return json.loads(cleaned)
    except Exception as e:
        return [{"error": str(e), "raw": response_text}]
    
genai.configure(api_key="AIzaSyC2CE4fX4Og7IxpHgIvjX3WbnXc_53kQas")
def call_gemini_response(prompt):
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(prompt)
    return clean_and_parse_json(response.text)

model = SentenceTransformer('all-MiniLM-L6-v2')

import pickle

# Caching để tiết kiệm thời gian encode
if os.path.exists("cached_skincare_df.pkl"):
    with open("cached_skincare_df.pkl", "rb") as f:
        skincare_df = pickle.load(f)
else:
    skincare_df = process_ingredients_column(skincare_df, model)
    with open("cached_skincare_df.pkl", "wb") as f:
        pickle.dump(skincare_df, f)



def retrieve_top_products(skin_type, concerns, df, ingredients=None, top_k=3):
    query_text = f"{skin_type} skin with concerns: {', '.join(concerns)}"
    query_vec = model.encode(query_text)
    sims = cosine_similarity([query_vec], list(df['embedding']))
    df = df.copy()
    df['sim_score'] = sims[0]

    top_df = df.sort_values(by='sim_score', ascending=False).head(100)

    if ingredients:
        ingredients = [i.lower().strip() for i in ingredients]

        def match_ingredients(ing_list):
            return any(
                any(target_ing in ing for ing in ing_list) for target_ing in ingredients
        )

        top_df = top_df[top_df['ingredient_list'].apply(match_ingredients)]

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



def recommend_products(skin_type, concerns, ingredients=None):
    prompt = build_prompt(skin_type, concerns, skincare_df)
    ai_response = call_gemini_response(prompt)
    grounded_products = retrieve_top_products(skin_type, concerns, skincare_df, ingredients=ingredients)
    return ai_response, grounded_products

from flask import Flask, jsonify, request, render_template
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        skin_type = request.form["skin_type"]
        concerns_input = request.form["concerns"]
        ingredients_input = request.form.get("ingredients", "")

        concerns = [c.strip().lower() for c in concerns_input.split(",") if c.strip()]
        ingredients = [i.strip().lower() for i in ingredients_input.split(",") if i.strip()]

        ai_result, rag_result_df = recommend_products(skin_type, concerns, ingredients)
        rag_result = rag_result_df[["product_name", "concerns", "ingredient_list"]].to_dict(orient="records")

        return render_template(
            "index.html",
            ai_result=ai_result,
            rag_result=rag_result,
            skin_type=skin_type,
            concerns=concerns_input,
            ingredients=ingredients_input
        )
    return render_template("index.html")



@app.route("/recommend", methods=["POST"])
def recommend_api():
    data = request.get_json()
    skin_type = data.get("skin_type", "")
    concerns = data.get("concerns", [])
    ingredients = data.get("ingredients", [])

    ai_result, grounded_products = recommend_products(skin_type, concerns, ingredients=ingredients)
    ingredient_results = search_multiple_ingredients(skincare_df, ingredients, top_k=3)

    if grounded_products.empty:
        grounded_result = []
    else:
        grounded_result = grounded_products[
            ['product_name', 'brand_name', 'concerns', 'ingredient_list', 'sim_score', 'total_score']
        ].to_dict(orient="records")

    return jsonify({
        "ai_result": ai_result,
        "grounded_result": grounded_result,
        "ingredient_results": ingredient_results
    })



if __name__ == "__main__":
    app.run(debug=True)
# python -m flask run