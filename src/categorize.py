"""Stage 3: give every product a category (the raw data has none).

Two methods, same output table `product_category(stockcode, category, method)`:

  rules   keyword rules, runs offline, instant, fully explainable.   <- default
  gemini  an LLM classifies product names in batches of 50.          <- needs GEMINI_API_KEY

Run:
  python -m src.categorize                  # rules
  python -m src.categorize --method gemini  # LLM (then compare accuracy, see evaluate())
  python -m src.categorize --make-labels    # writes outputs/label_me.csv (50 products to label by hand)
  python -m src.categorize --evaluate       # accuracy of the current table vs your hand labels
"""
import argparse
import json
import os
import re
import sqlite3
import pandas as pd
from src.config import DB_PATH, OUT_DIR

CATEGORIES = [
    "Christmas & seasonal", "Lighting & candles", "Jewellery & accessories",
    "Party & wrapping", "Stationery & paper", "Toys & games", "Garden & outdoor",
    "Bath & personal care", "Kitchen & dining", "Bags & storage", "Home decor",
    "Gifts & novelties",   # default bucket for anything the rules do not recognise
]

# Order matters: the FIRST matching rule wins (e.g. "PLAYING CARDS" must hit Toys before Stationery).
RULES = [
    ("Christmas & seasonal", r"CHRISTMAS|XMAS|SANTA|REINDEER|ADVENT|SNOWFLAKE|SNOWMAN|EASTER|HALLOWEEN|TINSEL|STOCKING|BAUBLE|WREATH|NATIVITY|ROBIN|GINGERBREAD|SNOW GLOBE|BUNNY|PUMPKIN|SKULL|VALENTINE"),
    ("Toys & games", r"PLAYING CARD|\bTOY|\bGAME|PUZZLE|DOLL|TEDDY|SPACEBOY|DINOSAUR|DOMINO|SKITTLE|YOYO|\bKIDS?\b|CHILDREN|CHILD'?S|BABY|\bPLAY\b|BUILDING BLOCK|MONEY BOX|PIRATE|SUPERHERO|\bBALL\b|CLAY|COLOURING|SKIPPING|LADDERS|BINGO|PAINT SET|GLIDER|DRAUGHTS|CHESS"),
    ("Jewellery & accessories", r"NECKLACE|BRACELET|EARRING|\bRING\b|BROOCH|PENDANT|\bBEADS?\b|CHARM|HAIR|SCARF|SUNGLASS|\bGLOVES?\b|\bPURSE|WALLET|\bHAT\b|UMBRELLA|KEYRING|KEY RING|ANKLET|TIARA|BANGLE|CUFFLINK|NAIL|MAKE UP|MAKEUP|COSMETIC"),
    ("Lighting & candles", r"CANDLE|LIGHT|LANTERN|LAMP|INCENSE|TEALIGHT|TEA LIGHT|NIGHTLIGHT|OIL BURNER|WICK|FLAME|CHANDELIER"),
    ("Party & wrapping", r"PARTY|WRAP|BUNTING|BALLOON|BIRTHDAY|RIBBON|TISSUE|CONFETTI|STREAMER|CRACKER|CAKE CASE|NAPKIN|PAPER CHAIN|PAPER PLATE|GIFT BAG|GIFT TAG|\bTAGS?\b|GARLAND|PINATA|BANNER"),
    ("Stationery & paper", r"\bCARDS?\b|NOTEBOOK|PENCIL|\bPENS?\b|STICKER|ENVELOPE|STATIONERY|\bPAPER\b|JOURNAL|DIARY|ERASER|RULER|STAMP|CALENDAR|\bBOOK|POSTCARD|BOOKMARK|SHARPENER|CRAYON|NOTEPAD|JIGSAW|CLIPBOARD|PAPERWEIGHT|LETTER"),
    ("Garden & outdoor", r"GARDEN|PLANT|FLOWER POT|\bSEED|WATERING|BIRD ?HOUSE|BIRD FEEDER|FEEDER|PARASOL|WINDMILL|BUCKET|TROWEL|OUTDOOR|HERB|GNOME|SCARECROW|LAWN|PLANTER|WELLINGTON|WINDCHIME|WIND CHIME|\bSPADE|\bFORK\b.*GARDEN|BIRD TABLE|HANGING BASKET"),
    ("Bath & personal care", r"SOAP|\bBATH|SPONGE|TOOTHBRUSH|LOTION|PERFUME|BODY|SHOWER|TOILET|FLANNEL|HOT WATER BOTTLE|SLIPPER|DRESSING GOWN|SPA\b|BEAUTY"),
    ("Bags & storage", r"\bBAGS?\b|STORAGE|BASKET|\bBOX\b|\bBOXES\b|TRUNK|\bCASES?\b|\bCHEST\b|HOLDALL|\bTIN\b|\bTINS\b|DRAWER|CABINET|ORGANI[SZ]ER|SUITCASE|LUGGAGE|SHOPPER|RUCKSACK|BACKPACK|\bPOUCH|WASHBAG|WASH BAG|LUNCH|CRATE"),
    ("Kitchen & dining", r"\bMUGS?\b|\bCUPS?\b|\bTEA\b|TEAPOT|TEACUP|\bPLATES?\b|\bBOWLS?\b|CUTLERY|SPOON|KNIFE|KITCHEN|\bCAKE|BAKING|\bJAM\b|\bEGG|\bSALT|COASTER|\bTRAY|BOTTLE|\bJUG\b|APRON|PICNIC|PANTRY|TEA ?TOWEL|OVEN|MEASURING|BREAD|BISCUIT|COOKIE|DINNER|SANDWICH|CORK|\bJAR\b|\bJARS\b|\bGLASSES\b|TUMBLER|WINE|SALAD|\bTEASPOON|PLACEMAT|\bPOT\b|COLANDER|SIEVE|RECIPE|CHOCOLATE|SWEET JAR|FOOD"),
    ("Home decor", r"CUSHION|MIRROR|FRAME|CLOCK|\bWALL\b|\bSIGN\b|DOORMAT|\bHOOK|\bKEY\b|CURTAIN|\bRUG\b|\bVASE|ORNAMENT|SHELF|\bTILE|BLANKET|\bCOVER\b|PHOTO|PICTURE|\bDOOR\b|WINDOW|STOOL|CHAIR|\bTABLE\b|LAMPSHADE|HANGER|DECORATION|\bSTAR\b|\bBIRD\b|BUTTERFLY|HEART|\bANGEL|BELL|FIGURE|STATUE|CANVAS|PLAQUE|MAGNET|WALLPAPER|PANEL|SCREEN|DOORSTOP|STAND\b|TRIVET|HOLDER|ARTIFICIAL|FLOWER|ROSE|\bBEAD CURTAIN|FELT|DOILY|CROCHET|BOARD|MEMO|CHALK|\bRACK|\bPEGS?\b|SILK FAN"),
]
_COMPILED = [(cat, re.compile(rx)) for cat, rx in RULES]


def rule_category(description: str) -> str:
    d = (description or "").upper()
    for cat, rx in _COMPILED:
        if rx.search(d):
            return cat
    return "Gifts & novelties"


# ---------------------------------------------------------------- LLM method (optional)
def _prompt(batch):
    cats = "\n".join(f"- {c}" for c in CATEGORIES)
    items = "\n".join(f"{i}. {d}" for i, d in enumerate(batch))
    return (
        "You classify product names of a UK gift-ware online retailer into ONE category each.\n"
        f"Allowed categories (use the exact text):\n{cats}\n\n"
        "Return ONLY a JSON object mapping each item number (as a string) to its category. "
        "No explanation, no markdown fences.\n\nItems:\n" + items
    )


def parse_llm_json(text: str, batch_len: int) -> dict:
    """Turn the model's reply into {index: category}; anything invalid becomes the default bucket."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = {}
    out = {}
    for i in range(batch_len):
        c = raw.get(str(i))
        out[i] = c if c in CATEGORIES else None   # None -> caller falls back to rules
    return out


def llm_categories(descriptions, batch_size=50):
    """Classify with Gemini. Needs `pip install google-genai` and GEMINI_API_KEY in .env."""
    from dotenv import load_dotenv
    from google import genai
    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")   # check the current model name
    result = []
    for start in range(0, len(descriptions), batch_size):
        batch = descriptions[start:start + batch_size]
        resp = client.models.generate_content(model=model, contents=_prompt(batch))
        parsed = parse_llm_json(resp.text, len(batch))
        for i, d in enumerate(batch):
            result.append(parsed[i] or rule_category(d))
        print(f"  classified {min(start + batch_size, len(descriptions))}/{len(descriptions)}")
    return result


# ---------------------------------------------------------------- build / evaluate
def build(method="rules"):
    con = sqlite3.connect(DB_PATH)
    prod = pd.read_sql_query("SELECT stockcode, description FROM products", con)
    if method == "gemini":
        prod["category"] = llm_categories(prod["description"].tolist())
    else:
        prod["category"] = prod["description"].map(rule_category)
    prod["method"] = method
    prod[["stockcode", "category", "method"]].to_sql("product_category", con, if_exists="replace", index=False)
    con.commit()
    share = pd.read_sql_query("""
        SELECT c.category, COUNT(DISTINCT c.stockcode) AS products,
               ROUND(SUM(s.revenue), 0) AS revenue,
               ROUND(100.0 * SUM(s.revenue) / (SELECT SUM(revenue) FROM sales), 1) AS revenue_share_pct
        FROM product_category c JOIN sales s USING (stockcode)
        GROUP BY c.category ORDER BY revenue DESC""", con)
    con.close()
    share.to_csv(OUT_DIR / "category_summary.csv", index=False)
    print(share.to_string(index=False))
    return share


def make_label_template(n=50, seed=7):
    con = sqlite3.connect(DB_PATH)
    prod = pd.read_sql_query("SELECT stockcode, description FROM products", con)
    con.close()
    sample = prod.sample(n, random_state=seed).copy()
    sample["my_label"] = ""
    sample.to_csv(OUT_DIR / "label_me.csv", index=False)
    print(f"Wrote {OUT_DIR/'label_me.csv'}.\nFill the my_label column using ONLY these categories:\n  " + "\n  ".join(CATEGORIES))


def evaluate():
    labels = pd.read_csv(OUT_DIR / "label_me.csv").dropna(subset=["my_label"])
    labels = labels[labels["my_label"].str.strip() != ""]
    con = sqlite3.connect(DB_PATH)
    cat = pd.read_sql_query("SELECT stockcode, category, method FROM product_category", con)
    con.close()
    m = labels.merge(cat, on="stockcode")
    acc = (m["my_label"].str.strip() == m["category"]).mean()
    print(f"method={m['method'].iloc[0]}  accuracy on {len(m)} hand-labelled products: {acc:.0%}")
    print(m[m["my_label"].str.strip() != m["category"]][["description", "my_label", "category"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["rules", "gemini"], default="rules")
    ap.add_argument("--make-labels", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    if a.make_labels:
        make_label_template()
    elif a.evaluate:
        evaluate()
    else:
        build(a.method)
