import pandas as pd
import os
import re
import pickle
from app.services.sentiment import enrich_sentiment_scores
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
PRODUCT_PATH = os.path.join(DATA_DIR, "product_info.csv")
REVIEW_PATHS = [os.path.join(DATA_DIR, "Reviews_1250-end.csv")]
MAPPING_PATH = os.path.join(DATA_DIR, "G:\language\Project2\TMG\ingredient_to_concern_mapping.txt")

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

def clean_ingredient_text(raw_text):
    if not isinstance(raw_text, str):
        return []
    text = raw_text.lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    items = [re.sub(r'[^a-z0-9 ]', '', i.strip()) for i in text.split(',') if i.strip()]
    return items

def map_ingredients_to_concerns(ingredient_list, ingredient_to_concern):
    matched = set()
    for ing in ingredient_list:
        for known_ing in ingredient_to_concern:
            if known_ing in ing:
                matched.add(ingredient_to_concern[known_ing])
    return list(matched) if matched else ['unknown']

model = SentenceTransformer('all-MiniLM-L6-v2')
ingredient_to_concern = load_ingredient_concern_mapping(MAPPING_PATH)

if os.path.exists("cached_skincare_df.pkl"):
    with open("cached_skincare_df.pkl", "rb") as f:
        skincare_df = pickle.load(f)
else:
    product_df = pd.read_csv(PRODUCT_PATH)
    review_df = pd.concat([pd.read_csv(p) for p in REVIEW_PATHS], ignore_index=True)
    skincare_df = review_df.merge(product_df, on='product_id', how='inner')

    for col in skincare_df.columns:
        if col.endswith('_x') and col[:-2] + '_y' in skincare_df.columns:
            base = col[:-2]
            skincare_df[base] = skincare_df[col].combine_first(skincare_df[base + '_y'])
            skincare_df.drop(columns=[col, base + '_y'], inplace=True)

    columns_to_keep = [
        'product_id', 'product_name', 'brand_name', 'ingredients', 'price_usd',
        'primary_category', 'skin_type', 'highlights', 'review_title',
        'is_recommended', 'rating'
    ]
    skincare_df = skincare_df[columns_to_keep]
    skincare_df = skincare_df[skincare_df['primary_category'].str.lower().str.contains('skincare', na=False)]
    skincare_df = skincare_df.dropna(subset=columns_to_keep)

    # sentiment + normalize
    skincare_df = enrich_sentiment_scores(skincare_df)

    # mapping concerns
    skincare_df['ingredient_list'] = skincare_df['ingredients'].apply(clean_ingredient_text)
    skincare_df['concerns'] = skincare_df['ingredient_list'].apply(
        lambda lst: map_ingredients_to_concerns(lst, ingredient_to_concern)
    )
    embeddings = model.encode(skincare_df['concerns'].astype(str).tolist(), batch_size=32, show_progress_bar=True)
    skincare_df['embedding'] = embeddings.tolist()

    with open("cached_skincare_df.pkl", "wb") as f:
        pickle.dump(skincare_df, f)
