def build_prompt(skin_type, concerns, product_df, ingredients=None):
    import random
    ingredients = ingredients or []
    product_examples = product_df.sample(3, random_state=42).to_dict(orient='records')
    examples_str = "\n".join([
        f"{i+1}. {{\"product_name\": \"{p['product_name']}\", \"brand_name\": \"{p['brand_name']}\", \"skin_type\": \"{p['skin_type']}\", \"concerns\": \"{p['concerns']}\", \"ingredients\": \"{p['ingredients']}\", \"price\": \"{p['price_usd']}\"}}"
        for i, p in enumerate(product_examples)
    ])

    prompt = f"""
You are a skincare expert. Recommend 3 suitable products in JSON format for someone with {skin_type} skin and concerns: {', '.join(concerns)}.
"""
    if ingredients:
        prompt += f"\nOnly include products that contain the following exactly ingredients that user choose(exactly spelling): {', '.join(ingredients)}."

    prompt += f"""

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