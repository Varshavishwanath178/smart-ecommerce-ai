from io import BytesIO

import speech_recognition as sr
import streamlit as st
import soundfile as sf
import pandas as pd
from gtts import gTTS
from gtts.tts import gTTSError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Smart E-Commerce AI & Voice Assistant",
    page_icon="🛍️",
    layout="wide",
)

PRODUCTS = [
    {
        "name": "Nova X Pro Smartphone",
        "category": "Electronics",
        "description": "5G smartphone with AMOLED display, fast processor, long battery life, and a high-resolution camera.",
        "price": 54999,
        "icon": "📱",
    },
    {
        "name": "PixelView 4K Smart TV",
        "category": "Electronics",
        "description": "Ultra HD 4K smart television with HDR picture, streaming apps, and immersive Dolby audio.",
        "price": 72999,
        "icon": "📺",
    },
    {
        "name": "SonicBeat Wireless Headphones",
        "category": "Electronics",
        "description": "Wireless over-ear headphones with active noise cancellation, rich bass, and all-day battery life.",
        "price": 8999,
        "icon": "🎧",
    },
    {
        "name": "ThunderMax Bluetooth Speaker",
        "category": "Electronics",
        "description": "Portable Bluetooth speaker with powerful bass, clear sound, and a rugged water-resistant body.",
        "price": 4999,
        "icon": "🔊",
    },
    {
        "name": "Everyday Cotton Kurta",
        "category": "Fashion & Apparel",
        "description": "Soft breathable cotton kurta with a comfortable regular fit, printed design, and easy-care fabric.",
        "price": 1499,
        "icon": "👘",
    },
    {
        "name": "Classic Denim Jacket",
        "category": "Fashion & Apparel",
        "description": "Versatile cotton denim jacket with durable stitching, button closure, and a timeless casual style.",
        "price": 2799,
        "icon": "🧥",
    },
    {
        "name": "FlexRun Running Shoes",
        "category": "Fashion & Apparel",
        "description": "Lightweight everyday running shoes with cushioned soles, breathable mesh, and a flexible grip.",
        "price": 3499,
        "icon": "👟",
    },
    {
        "name": "Everyday Canvas Backpack",
        "category": "Fashion & Apparel",
        "description": "Durable canvas backpack with padded laptop compartment, adjustable straps, and roomy storage.",
        "price": 1899,
        "icon": "🎒",
    },
    {
        "name": "BrewMate French Press",
        "category": "Home & Kitchen",
        "description": "Heat-resistant glass coffee maker with stainless steel filter for rich, freshly brewed coffee.",
        "price": 1299,
        "icon": "☕",
    },
    {
        "name": "ChefPro Nonstick Cookware Set",
        "category": "Home & Kitchen",
        "description": "Kitchen cookware set with nonstick coating, sturdy handles, and easy-to-clean pots and pans.",
        "price": 3999,
        "icon": "🍳",
    },
    {
        "name": "CozyNest Cotton Bedsheet",
        "category": "Home & Kitchen",
        "description": "Soft breathable cotton bedsheet set with fade-resistant colors and matching pillow covers.",
        "price": 1799,
        "icon": "🛏️",
    },
    {
        "name": "GlowPure Vitamin C Face Serum",
        "category": "Beauty & Personal Care",
        "description": "Lightweight vitamin C facial serum for daily skincare with a hydrating, fast-absorbing formula.",
        "price": 699,
        "icon": "🧴",
    },
    {
        "name": "SilkShine Hair Care Kit",
        "category": "Beauty & Personal Care",
        "description": "Nourishing shampoo and conditioner hair care kit for smooth, shiny hair and everyday use.",
        "price": 899,
        "icon": "🧖",
    },
    {
        "name": "FreshBloom Eau de Parfum",
        "category": "Beauty & Personal Care",
        "description": "Long-lasting floral perfume with fresh citrus notes, presented in a travel-friendly glass bottle.",
        "price": 1599,
        "icon": "🌸",
    },
    {
        "name": "PulseFit Smartwatch",
        "category": "Sports & Fitness",
        "description": "Fitness smartwatch with heart-rate tracking, multiple workout modes, notifications, and water resistance.",
        "price": 6499,
        "icon": "⌚",
    },
    {
        "name": "PowerFlex Yoga Mat",
        "category": "Sports & Fitness",
        "description": "Non-slip cushioned yoga mat for home workouts, stretching, meditation, and daily fitness practice.",
        "price": 1199,
        "icon": "🧘",
    },
    {
        "name": "ProGrip Adjustable Dumbbells",
        "category": "Sports & Fitness",
        "description": "Compact adjustable dumbbell set for strength training, home gym workouts, and full-body exercise.",
        "price": 4999,
        "icon": "🏋️",
    },
    {
        "name": "The Quiet Horizon",
        "category": "Books",
        "description": "An uplifting contemporary fiction novel about friendship, family, and finding a new beginning.",
        "price": 499,
        "icon": "📖",
    },
    {
        "name": "Everyday Personal Finance",
        "category": "Books",
        "description": "A practical personal finance guide covering budgeting, saving, investing, and building healthy money habits.",
        "price": 599,
        "icon": "📚",
    },
    {
        "name": "The Curious Science Handbook",
        "category": "Books",
        "description": "An illustrated science book exploring space, nature, physics, and fascinating everyday discoveries.",
        "price": 799,
        "icon": "🔭",
    },
]

GENUINE_REVIEWS = [
    "The phone arrived in two days and the camera takes sharp photos in daylight.",
    "Good sound quality for the price. The headband feels comfortable after a few hours.",
    "Setup was simple and the picture looks clear. The remote is a little slow sometimes.",
    "Battery lasts most of my workday with brightness turned down.",
    "I have used this speaker for a month. It is loud enough for a small room.",
    "The watch tracks my walks accurately, although the strap took a day to get used to.",
    "Solid laptop for everyday work. The keyboard is comfortable and it starts quickly.",
    "The power bank charged my phone twice. It is heavier than I expected.",
    "Nice camera and good image quality. Autofocus is fast for family photos.",
    "Works as described. Delivery was on time and the packaging was intact.",
    "The screen is bright and apps load quickly. Overall, good value for money.",
    "Decent product. I wish the charging cable were a little longer.",
]

SUSPICIOUS_REVIEWS = [
    "Best product ever!!! Buy now!!! You will not regret it!!!",
    "Amazing amazing amazing. Five stars. Best deal. Buy now.",
    "This is the number one product in the world, guaranteed perfect!!!",
    "Click here to get the best price. Limited offer, order immediately!",
    "Perfect product, perfect seller, perfect delivery, absolutely no flaws.",
    "Unbelievable deal!!! The best ever!!! Everyone must buy this right now.",
    "I got this free and was told to leave five stars. Best product ever.",
    "Amazing quality!!! 100 percent authentic!!! Buy today before it is gone!!!",
    "Best product best product best product. Five stars and highly recommended.",
    "You must purchase this item now. Guaranteed results and the lowest price!",
    "Totally perfect in every way. No issues at all. Buy now!!!",
    "Fantastic!!! Excellent!!! Incredible!!! Best deal online!!!",
]

LANGUAGES = {
    "English": {"speech": "en-IN", "tts": "en"},
    "Hindi": {"speech": "hi-IN", "tts": "hi"},
    "Kannada": {"speech": "kn-IN", "tts": "kn"},
    "Tamil": {"speech": "ta-IN", "tts": "ta"},
    "Telugu": {"speech": "te-IN", "tts": "te"},
    "Malayalam": {"speech": "ml-IN", "tts": "ml"},
    "Marathi": {"speech": "mr-IN", "tts": "mr"},
    "Bengali": {"speech": "bn-IN", "tts": "bn"},
    "Gujarati": {"speech": "gu-IN", "tts": "gu"},
    "Punjabi": {"speech": "pa-IN", "tts": "pa"},
    "Urdu": {"speech": "ur-IN", "tts": "ur"},
    "Spanish": {"speech": "es-ES", "tts": "es"},
    "French": {"speech": "fr-FR", "tts": "fr"},
    "German": {"speech": "de-DE", "tts": "de"},
    "Japanese": {"speech": "ja-JP", "tts": "ja"},
    "Arabic": {"speech": "ar-SA", "tts": "ar"},
}

INTENT_KEYWORDS = {
    "en": {
        "greeting": ("hello", "hey", "good morning", "good evening"),
        "shipping": ("shipping", "delivery", "deliver", "ship", "arrive", "order status"),
        "returns": ("return", "refund", "exchange", "replace"),
        "payments": ("upi", "payment", "pay", "card", "credit", "debit", "net banking"),
        "warranty": ("warranty", "guarantee", "repair"),
    },
    "hi": {
        "greeting": ("नमस्ते", "नमस्कार", "प्रणाम", "नमस्ते जी"),
        "shipping": ("डिलीवरी", "शिपिंग", "पहुंचेगा", "पहुंचेगी", "पार्सल", "ऑर्डर कहां"),
        "returns": ("वापसी", "वापस", "रिटर्न", "रिफंड", "बदलना"),
        "payments": ("भुगतान", "पेमेंट", "यूपीआई", "upi", "कार्ड", "पैसे"),
        "warranty": ("वारंटी", "गारंटी", "मरम्मत"),
    },
    "kn": {
        "greeting": ("ನಮಸ್ಕಾರ", "ಹಲೋ", "ಶುಭೋದಯ"),
        "shipping": ("ವಿತರಣೆ", "ಡೆಲಿವರಿ", "ತಲುಪುತ್ತದೆ", "ತಲುಪಲು", "ಆರ್ಡರ್ ಎಲ್ಲಿದೆ", "ಕಳುಹಿಸುವ"),
        "returns": ("ಹಿಂತಿರುಗಿ", "ವಾಪಸ್", "ಮರುಪಾವತಿ", "ಹಣ ವಾಪಸ್", "ಬದಲಾಯಿಸಿ"),
        "payments": ("ಪಾವತಿ", "ಯುಪಿಐ", "ಕಾರ್ಡ್", "ಹಣ ಪಾವತಿ", "ಡೆಬಿಟ್", "ಕ್ರೆಡಿಟ್"),
        "warranty": ("ಖಾತರಿ", "ವಾರಂಟಿ", "ದುರಸ್ತಿ"),
    },
    "ta": {
        "greeting": ("வணக்கம்", "நலமா"),
        "shipping": ("டெலிவரி", "விநியோகம்", "எப்போது வரும்", "அனுப்ப", "ஆர்டர் எங்கே"),
        "returns": ("திரும்ப", "பணத்தைத் திரும்ப", "பணத்தை திரும்ப", "திருப்பி", "மாற்ற"),
        "payments": ("பணம் செலுத்த", "கட்டணம்", "யுபிஐ", "கார்டு", "கிரெடிட்", "டெபிட்"),
        "warranty": ("உத்தரவாதம்", "வாரண்டி", "பழுது"),
    },
    "te": {
        "greeting": ("నమస్కారం", "హలో", "శుభోదయం"),
        "shipping": ("డెలివరీ", "రవాణా", "ఎప్పుడు వస్తుంది", "చేరుతుంది", "ఆర్డర్ ఎక్కడ"),
        "returns": ("తిరిగి", "రిటర్న్", "వాపసు", "రీఫండ్", "మార్చు"),
        "payments": ("చెల్లింపు", "యూపీఐ", "కార్డు", "క్రెడిట్", "డెబిట్", "డబ్బు చెల్లించు"),
        "warranty": ("వారంటీ", "హామీ", "మరమ్మత్తు"),
    },
    "ml": {
        "greeting": ("നമസ്കാരം", "ഹലോ", "സുപ്രഭാതം"),
        "shipping": ("ഡെലിവറി", "വിതരണം", "എപ്പോൾ എത്തും", "എത്തും", "ഓർഡർ എവിടെ"),
        "returns": ("തിരികെ", "റിട്ടേൺ", "റീഫണ്ട്", "പണം തിരികെ", "മാറ്റുക"),
        "payments": ("പേയ്മെന്റ്", "പണം അടയ്ക്ക", "യുപിഐ", "കാർഡ്", "ക്രെഡിറ്റ്", "ഡെബിറ്റ്"),
        "warranty": ("വാറന്റി", "ഗ്യാരണ്ടി", "അറ്റകുറ്റപ്പണി"),
    },
    "mr": {
        "greeting": ("नमस्कार", "नमस्ते", "शुभ सकाळ"),
        "shipping": ("डिलिव्हरी", "वितरण", "कधी येईल", "पोहोचेल", "ऑर्डर कुठे"),
        "returns": ("परतावा", "परत करा", "रिफंड", "बदला", "विनिमय"),
        "payments": ("पेमेंट", "पैसे भरा", "यूपीआय", "कार्ड", "क्रेडिट", "डेबिट"),
        "warranty": ("वॉरंटी", "हमी", "दुरुस्ती"),
    },
    "bn": {
        "greeting": ("নমস্কার", "নমস্তে", "হ্যালো", "সুপ্রভাত"),
        "shipping": ("ডেলিভারি", "সরবরাহ", "কবে আসবে", "পৌঁছাবে", "অর্ডার কোথায়"),
        "returns": ("ফেরত", "রিটার্ন", "রিফান্ড", "টাকা ফেরত", "বদল"),
        "payments": ("পেমেন্ট", "টাকা দিন", "ইউপিআই", "কার্ড", "ক্রেডিট", "ডেবিট"),
        "warranty": ("ওয়ারেন্টি", "গ্যারান্টি", "মেরামত"),
    },
    "gu": {
        "greeting": ("નમસ્તે", "નમસ્કાર", "સુપ્રભાત"),
        "shipping": ("ડિલિવરી", "વિતરણ", "ક્યારે આવશે", "પહોંચશે", "ઓર્ડર ક્યાં"),
        "returns": ("પાછું", "પરત", "રિફંડ", "પૈસા પાછા", "બદલો"),
        "payments": ("ચુકવણી", "પેમેન્ટ", "યુપીઆઈ", "કાર્ડ", "ક્રેડિટ", "ડેબિટ"),
        "warranty": ("વોરંટી", "ગેરંટી", "સમારકામ"),
    },
    "pa": {
        "greeting": ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "ਨਮਸਤੇ", "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ"),
        "shipping": ("ਡਿਲਿਵਰੀ", "ਡਿਲੀਵਰੀ", "ਕਦੋਂ ਆਵੇਗਾ", "ਪਹੁੰਚੇਗਾ", "ਆਰਡਰ ਕਿੱਥੇ"),
        "returns": ("ਵਾਪਸ", "ਰਿਫੰਡ", "ਪੈਸੇ ਵਾਪਸ", "ਬਦਲੋ", "ਮੋੜੋ"),
        "payments": ("ਭੁਗਤਾਨ", "ਪੇਮੈਂਟ", "ਯੂਪੀਆਈ", "ਕਾਰਡ", "ਕ੍ਰੈਡਿਟ", "ਡੈਬਿਟ"),
        "warranty": ("ਵਾਰੰਟੀ", "ਗਾਰੰਟੀ", "ਮੁਰੰਮਤ"),
    },
    "ur": {
        "greeting": ("السلام علیکم", "ہیلو", "آداب"),
        "shipping": ("ترسیل", "ڈیلیوری", "کب پہنچے گا", "پارسل", "آرڈر کہاں"),
        "returns": ("واپسی", "واپس", "رقم واپس", "ریفنڈ", "تبدیل"),
        "payments": ("ادائیگی", "یو پی آئی", "یوپی آئی", "کارڈ", "کریڈٹ", "ڈیبٹ"),
        "warranty": ("وارنٹی", "ضمانت", "مرمت"),
    },
    "es": {
        "greeting": ("hola", "buenos días", "buenas tardes"),
        "shipping": ("envío", "envios", "entrega", "pedido", "llegará", "llega"),
        "returns": ("devolución", "devoluciones", "reembolso", "devolver", "cambiar"),
        "payments": ("pago", "pagar", "upi", "tarjeta", "crédito", "débito"),
        "warranty": ("garantía", "reparación"),
    },
    "fr": {
        "greeting": ("bonjour", "salut", "bonsoir"),
        "shipping": ("livraison", "expédition", "colis", "commande", "arrivera"),
        "returns": ("retour", "retours", "remboursement", "échanger", "renvoyer"),
        "payments": ("paiement", "payer", "upi", "carte", "crédit", "débit"),
        "warranty": ("garantie", "réparation"),
    },
    "de": {
        "greeting": ("hallo", "guten morgen", "guten tag", "guten abend"),
        "shipping": ("versand", "lieferung", "paket", "bestellung", "ankommen"),
        "returns": ("rückgabe", "rückerstattung", "zurückgeben", "umtausch"),
        "payments": ("zahlung", "bezahlen", "upi", "karte", "kredit", "debit"),
        "warranty": ("garantie", "gewährleistung", "reparatur"),
    },
    "ja": {
        "greeting": ("こんにちは", "おはよう", "こんばんは"),
        "shipping": ("配送", "配達", "発送", "いつ届く", "注文はどこ"),
        "returns": ("返品", "返金", "交換"),
        "payments": ("支払い", "決済", "upi", "カード", "クレジット", "デビット"),
        "warranty": ("保証", "修理"),
    },
    "ar": {
        "greeting": ("مرحبا", "السلام عليكم", "أهلا", "أهلًا"),
        "shipping": ("الشحن", "التوصيل", "متى يصل", "الطلب", "طرد"),
        "returns": ("إرجاع", "استرجاع", "استرداد", "إعادة", "تبديل"),
        "payments": ("الدفع", "ادفع", "يو بي آي", "بطاقة", "ائتمان", "خصم"),
        "warranty": ("الضمان", "كفالة", "إصلاح"),
    },
}

CHATBOT_RESPONSES = {
    "en": {
        "greeting": "Hello! I can help with shipping, returns, payments, and warranties. What would you like to know?",
        "shipping": "We offer free standard shipping on orders over ₹999. Most orders arrive within 3–5 business days. You can track your order from your account.",
        "returns": "Eligible items can be returned or exchanged within 7 days of delivery, in their original condition and packaging. Contact support to get started.",
        "payments": "We accept UPI, credit and debit cards, net banking, and cash on delivery where available. Payments are processed securely at checkout.",
        "warranty": "Most electronics include a 1-year manufacturer warranty. Check the product listing and warranty card for coverage details.",
        "general": "I can help with shipping and delivery, returns and refunds, UPI or card payments, and product warranties. What would you like to know?",
    },
    "hi": {
        "greeting": "नमस्ते! मैं डिलीवरी, वापसी, भुगतान और वारंटी में मदद कर सकता हूँ। आप क्या जानना चाहते हैं?",
        "shipping": "₹999 से अधिक के ऑर्डर पर मानक डिलीवरी मुफ़्त है। अधिकतर ऑर्डर 3–5 कार्यदिवस में पहुँचते हैं। आप अपने खाते में ऑर्डर ट्रैक कर सकते हैं।",
        "returns": "पात्र सामान डिलीवरी के 7 दिनों के भीतर उसकी मूल स्थिति और पैकेजिंग में वापस या बदल सकते हैं। शुरुआत करने के लिए सहायता टीम से संपर्क करें।",
        "payments": "हम UPI, क्रेडिट और डेबिट कार्ड, नेट बैंकिंग और उपलब्धता के अनुसार कैश ऑन डिलीवरी स्वीकार करते हैं। भुगतान सुरक्षित रूप से किया जाता है।",
        "warranty": "अधिकतर इलेक्ट्रॉनिक्स पर निर्माता की 1 साल की वारंटी होती है। कवरेज के लिए उत्पाद पेज और वारंटी कार्ड देखें।",
        "general": "मैं डिलीवरी, वापसी और रिफंड, UPI या कार्ड भुगतान और उत्पाद वारंटी में मदद कर सकता हूँ। आप क्या जानना चाहते हैं?",
    },
    "kn": {
        "greeting": "ನಮಸ್ಕಾರ! ವಿತರಣೆ, ಹಿಂತಿರುಗಿಸುವಿಕೆ, ಪಾವತಿ ಮತ್ತು ಖಾತರಿ ಕುರಿತು ನಾನು ಸಹಾಯ ಮಾಡುತ್ತೇನೆ. ನಿಮಗೆ ಏನು ತಿಳಿಯಬೇಕು?",
        "shipping": "₹999 ಕ್ಕಿಂತ ಹೆಚ್ಚಿನ ಆರ್ಡರ್‌ಗಳಿಗೆ ಸಾಮಾನ್ಯ ವಿತರಣೆ ಉಚಿತ. ಹೆಚ್ಚಿನ ಆರ್ಡರ್‌ಗಳು 3–5 ಕೆಲಸದ ದಿನಗಳಲ್ಲಿ ತಲುಪುತ್ತವೆ. ನಿಮ್ಮ ಖಾತೆಯಲ್ಲಿ ಆರ್ಡರ್ ಅನ್ನು ಟ್ರ್ಯಾಕ್ ಮಾಡಬಹುದು.",
        "returns": "ಅರ್ಹ ವಸ್ತುಗಳನ್ನು ವಿತರಣೆಯ 7 ದಿನಗಳೊಳಗೆ ಮೂಲ ಸ್ಥಿತಿ ಮತ್ತು ಪ್ಯಾಕೇಜಿಂಗ್‌ನೊಂದಿಗೆ ಹಿಂತಿರುಗಿಸಬಹುದು ಅಥವಾ ಬದಲಾಯಿಸಬಹುದು. ಸಹಾಯಕ್ಕಾಗಿ ಬೆಂಬಲ ತಂಡವನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "payments": "ನಾವು UPI, ಕ್ರೆಡಿಟ್ ಮತ್ತು ಡೆಬಿಟ್ ಕಾರ್ಡ್, ನೆಟ್ ಬ್ಯಾಂಕಿಂಗ್ ಮತ್ತು ಲಭ್ಯವಿದ್ದಲ್ಲಿ ಕ್ಯಾಶ್ ಆನ್ ಡೆಲಿವರಿಯನ್ನು ಸ್ವೀಕರಿಸುತ್ತೇವೆ. ಪಾವತಿಗಳು ಸುರಕ್ಷಿತವಾಗಿವೆ.",
        "warranty": "ಹೆಚ್ಚಿನ ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್‌ಗೆ ತಯಾರಕರ 1 ವರ್ಷದ ಖಾತರಿ ಇರುತ್ತದೆ. ವಿವರಗಳಿಗಾಗಿ ಉತ್ಪನ್ನ ಪಟ್ಟಿ ಮತ್ತು ಖಾತರಿ ಕಾರ್ಡ್ ಪರಿಶೀಲಿಸಿ.",
        "general": "ವಿತರಣೆ, ಹಿಂತಿರುಗಿಸುವಿಕೆ ಮತ್ತು ಮರುಪಾವತಿ, UPI ಅಥವಾ ಕಾರ್ಡ್ ಪಾವತಿ ಹಾಗೂ ಉತ್ಪನ್ನ ಖಾತರಿ ಕುರಿತು ನಾನು ಸಹಾಯ ಮಾಡುತ್ತೇನೆ. ನಿಮಗೆ ಏನು ತಿಳಿಯಬೇಕು?",
    },
    "ta": {
        "greeting": "வணக்கம்! விநியோகம், திரும்பப்பெறுதல், கட்டணம் மற்றும் உத்தரவாதம் குறித்து உதவுகிறேன். என்ன தெரிந்துகொள்ள விரும்புகிறீர்கள்?",
        "shipping": "₹999-க்கு மேற்பட்ட ஆர்டர்களுக்கு வழக்கமான விநியோகம் இலவசம். பெரும்பாலான ஆர்டர்கள் 3–5 வேலை நாட்களில் வந்து சேரும். உங்கள் கணக்கில் ஆர்டரைக் கண்காணிக்கலாம்.",
        "returns": "தகுதியுள்ள பொருட்களை விநியோகித்த 7 நாட்களுக்குள் அசல் நிலையில் மற்றும் பேக்கிங்குடன் திருப்பலாம் அல்லது மாற்றலாம். உதவிக்கு ஆதரவு குழுவைத் தொடர்புகொள்ளுங்கள்.",
        "payments": "UPI, கிரெடிட் மற்றும் டெபிட் கார்டுகள், நெட் பேங்கிங் மற்றும் கிடைக்கும் இடங்களில் பணம் செலுத்திப் பெறுதல் ஆகியவற்றை ஏற்கிறோம். கட்டணம் பாதுகாப்பானது.",
        "warranty": "பெரும்பாலான மின்னணுப் பொருட்களுக்கு உற்பத்தியாளரின் 1 ஆண்டு உத்தரவாதம் உண்டு. விவரங்களுக்கு தயாரிப்பு பட்டியலையும் உத்தரவாத அட்டையையும் பார்க்கவும்.",
        "general": "விநியோகம், திரும்பப்பெறுதல், UPI அல்லது கார்டு கட்டணம் மற்றும் தயாரிப்பு உத்தரவாதம் குறித்து உதவுகிறேன். என்ன தெரிந்துகொள்ள விரும்புகிறீர்கள்?",
    },
    "te": {
        "greeting": "నమస్కారం! డెలివరీ, తిరిగి ఇవ్వడం, చెల్లింపులు మరియు వారంటీ గురించి సహాయం చేస్తాను. మీకు ఏమి తెలుసుకోవాలి?",
        "shipping": "₹999 కంటే ఎక్కువ ఆర్డర్లకు సాధారణ డెలివరీ ఉచితం. చాలా ఆర్డర్లు 3–5 పని దినాల్లో చేరుతాయి. మీ ఖాతాలో ఆర్డర్‌ను ట్రాక్ చేయవచ్చు.",
        "returns": "అర్హమైన వస్తువులను డెలివరీ అయిన 7 రోజుల్లో అసలు స్థితి, ప్యాకేజింగ్‌తో తిరిగి ఇవ్వవచ్చు లేదా మార్చుకోవచ్చు. సహాయం కోసం సపోర్ట్‌ను సంప్రదించండి.",
        "payments": "UPI, క్రెడిట్ మరియు డెబిట్ కార్డులు, నెట్ బ్యాంకింగ్, అందుబాటులో ఉన్న చోట క్యాష్ ఆన్ డెలివరీని అంగీకరిస్తాం. చెల్లింపులు సురక్షితం.",
        "warranty": "చాలా ఎలక్ట్రానిక్స్‌కు తయారీదారు 1 సంవత్సరం వారంటీ ఉంటుంది. వివరాల కోసం ఉత్పత్తి జాబితా మరియు వారంటీ కార్డును చూడండి.",
        "general": "డెలివరీ, తిరిగి ఇవ్వడం మరియు రీఫండ్, UPI లేదా కార్డు చెల్లింపులు, ఉత్పత్తి వారంటీ గురించి సహాయం చేస్తాను. ఏమి తెలుసుకోవాలి?",
    },
    "ml": {
        "greeting": "നമസ്കാരം! ഡെലിവറി, മടക്കിനൽകൽ, പേയ്മെന്റ്, വാറന്റി എന്നിവയിൽ സഹായിക്കാം. എന്താണ് അറിയേണ്ടത്?",
        "shipping": "₹999-ൽ കൂടുതലുള്ള ഓർഡറുകൾക്ക് സാധാരണ ഡെലിവറി സൗജന്യമാണ്. മിക്ക ഓർഡറുകളും 3–5 പ്രവൃത്തി ദിവസങ്ങൾക്കുള്ളിൽ എത്തും. അക്കൗണ്ടിൽ ഓർഡർ ട്രാക്ക് ചെയ്യാം.",
        "returns": "യോഗ്യമായ ഉൽപ്പന്നങ്ങൾ ഡെലിവറി ചെയ്ത് 7 ദിവസത്തിനുള്ളിൽ യഥാർത്ഥ അവസ്ഥയിലും പാക്കിംഗിലും മടക്കുകയോ മാറ്റുകയോ ചെയ്യാം. സഹായത്തിന് പിന്തുണാ ടീമിനെ ബന്ധപ്പെടുക.",
        "payments": "UPI, ക്രെഡിറ്റ്, ഡെബിറ്റ് കാർഡുകൾ, നെറ്റ് ബാങ്കിംഗ്, ലഭ്യമായിടത്ത് ക്യാഷ് ഓൺ ഡെലിവറി എന്നിവ സ്വീകരിക്കുന്നു. പേയ്മെന്റുകൾ സുരക്ഷിതമാണ്.",
        "warranty": "മിക്ക ഇലക്ട്രോണിക്സ് ഉൽപ്പന്നങ്ങൾക്കും നിർമ്മാതാവിന്റെ 1 വർഷത്തെ വാറന്റിയുണ്ട്. വിശദാംശങ്ങൾക്ക് ഉൽപ്പന്ന വിവരങ്ങളും വാറന്റി കാർഡും പരിശോധിക്കുക.",
        "general": "ഡെലിവറി, മടക്കിനൽകലും റീഫണ്ടും, UPI അല്ലെങ്കിൽ കാർഡ് പേയ്മെന്റും, ഉൽപ്പന്ന വാറന്റിയും സംബന്ധിച്ച് സഹായിക്കാം. എന്താണ് അറിയേണ്ടത്?",
    },
    "mr": {
        "greeting": "नमस्कार! डिलिव्हरी, परतावा, पेमेंट आणि वॉरंटीबद्दल मी मदत करू शकतो. तुम्हाला काय जाणून घ्यायचे आहे?",
        "shipping": "₹999 पेक्षा जास्त ऑर्डरवर नियमित डिलिव्हरी मोफत आहे. बहुतेक ऑर्डर 3–5 कामकाजाच्या दिवसांत पोहोचतात. खात्यातून ऑर्डर ट्रॅक करा.",
        "returns": "पात्र वस्तू डिलिव्हरीनंतर 7 दिवसांत मूळ स्थितीत आणि पॅकेजिंगसह परत किंवा बदलता येतात. सुरुवात करण्यासाठी सहाय्य टीमशी संपर्क साधा.",
        "payments": "आम्ही UPI, क्रेडिट आणि डेबिट कार्ड, नेट बँकिंग आणि उपलब्ध ठिकाणी कॅश ऑन डिलिव्हरी स्वीकारतो. पेमेंट सुरक्षितपणे केले जाते.",
        "warranty": "बहुतेक इलेक्ट्रॉनिक्सवर उत्पादकाची 1 वर्षाची वॉरंटी असते. तपशीलांसाठी उत्पादनाची माहिती आणि वॉरंटी कार्ड पाहा.",
        "general": "डिलिव्हरी, परतावा आणि रिफंड, UPI किंवा कार्ड पेमेंट आणि उत्पादन वॉरंटीबद्दल मी मदत करू शकतो. काय जाणून घ्यायचे आहे?",
    },
    "bn": {
        "greeting": "নমস্কার! ডেলিভারি, ফেরত, পেমেন্ট এবং ওয়ারেন্টি সম্পর্কে সাহায্য করতে পারি। আপনি কী জানতে চান?",
        "shipping": "₹999-এর বেশি অর্ডারে সাধারণ ডেলিভারি বিনামূল্যে। বেশিরভাগ অর্ডার 3–5 কর্মদিবসের মধ্যে পৌঁছায়। আপনার অ্যাকাউন্ট থেকে অর্ডার ট্র্যাক করুন।",
        "returns": "যোগ্য পণ্য ডেলিভারির 7 দিনের মধ্যে আসল অবস্থা ও প্যাকেজিংসহ ফেরত বা বদল করা যায়। শুরু করতে সহায়তা দলের সঙ্গে যোগাযোগ করুন।",
        "payments": "আমরা UPI, ক্রেডিট ও ডেবিট কার্ড, নেট ব্যাংকিং এবং যেখানে উপলভ্য ক্যাশ অন ডেলিভারি গ্রহণ করি। পেমেন্ট নিরাপদে প্রক্রিয়া করা হয়।",
        "warranty": "বেশিরভাগ ইলেকট্রনিক পণ্যে প্রস্তুতকারকের 1 বছরের ওয়ারেন্টি থাকে। বিস্তারিত জানতে পণ্যের তালিকা ও ওয়ারেন্টি কার্ড দেখুন।",
        "general": "ডেলিভারি, ফেরত ও রিফান্ড, UPI বা কার্ড পেমেন্ট এবং পণ্যের ওয়ারেন্টি সম্পর্কে সাহায্য করতে পারি। কী জানতে চান?",
    },
    "gu": {
        "greeting": "નમસ્તે! ડિલિવરી, પરત, ચુકવણી અને વોરંટી વિશે હું મદદ કરી શકું છું. તમે શું જાણવા માંગો છો?",
        "shipping": "₹999થી વધુના ઓર્ડર પર સામાન્ય ડિલિવરી મફત છે. મોટાભાગના ઓર્ડર 3–5 કામકાજના દિવસોમાં પહોંચે છે. તમારા ખાતામાં ઓર્ડર ટ્રૅક કરો.",
        "returns": "પાત્ર વસ્તુઓ ડિલિવરીના 7 દિવસની અંદર મૂળ સ્થિતિ અને પેકિંગ સાથે પરત અથવા બદલી શકાય છે. શરૂઆત કરવા સહાય ટીમનો સંપર્ક કરો.",
        "payments": "અમે UPI, ક્રેડિટ અને ડેબિટ કાર્ડ, નેટ બેંકિંગ અને ઉપલબ્ધ હોય ત્યાં કેશ ઑન ડિલિવરી સ્વીકારીએ છીએ. ચુકવણી સુરક્ષિત છે.",
        "warranty": "મોટાભાગના ઇલેક્ટ્રોનિક્સ પર ઉત્પાદકની 1 વર્ષની વોરંટી હોય છે. વિગતો માટે ઉત્પાદનની યાદી અને વોરંટી કાર્ડ જુઓ.",
        "general": "ડિલિવરી, પરત અને રિફંડ, UPI અથવા કાર્ડ ચુકવણી અને ઉત્પાદન વોરંટી વિશે હું મદદ કરી શકું છું. શું જાણવું છે?",
    },
    "pa": {
        "greeting": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਡਿਲਿਵਰੀ, ਵਾਪਸੀ, ਭੁਗਤਾਨ ਅਤੇ ਵਾਰੰਟੀ ਬਾਰੇ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ। ਤੁਸੀਂ ਕੀ ਜਾਣਨਾ ਚਾਹੁੰਦੇ ਹੋ?",
        "shipping": "₹999 ਤੋਂ ਵੱਧ ਦੇ ਆਰਡਰਾਂ ਲਈ ਆਮ ਡਿਲਿਵਰੀ ਮੁਫ਼ਤ ਹੈ। ਜ਼ਿਆਦਾਤਰ ਆਰਡਰ 3–5 ਕੰਮਕਾਜੀ ਦਿਨਾਂ ਵਿੱਚ ਪਹੁੰਚਦੇ ਹਨ। ਆਪਣੇ ਖਾਤੇ ਵਿੱਚ ਆਰਡਰ ਟ੍ਰੈਕ ਕਰੋ।",
        "returns": "ਯੋਗ ਚੀਜ਼ਾਂ ਡਿਲਿਵਰੀ ਤੋਂ 7 ਦਿਨਾਂ ਅੰਦਰ ਅਸਲ ਹਾਲਤ ਅਤੇ ਪੈਕਿੰਗ ਸਮੇਤ ਵਾਪਸ ਜਾਂ ਬਦਲੀਆਂ ਜਾ ਸਕਦੀਆਂ ਹਨ। ਮਦਦ ਲਈ ਸਹਾਇਤਾ ਟੀਮ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
        "payments": "ਅਸੀਂ UPI, ਕ੍ਰੈਡਿਟ ਅਤੇ ਡੈਬਿਟ ਕਾਰਡ, ਨੈੱਟ ਬੈਂਕਿੰਗ ਅਤੇ ਜਿੱਥੇ ਉਪਲਬਧ ਹੋਵੇ ਕੈਸ਼ ਆਨ ਡਿਲਿਵਰੀ ਸਵੀਕਾਰ ਕਰਦੇ ਹਾਂ। ਭੁਗਤਾਨ ਸੁਰੱਖਿਅਤ ਹੈ।",
        "warranty": "ਜ਼ਿਆਦਾਤਰ ਇਲੈਕਟ੍ਰਾਨਿਕ ਚੀਜ਼ਾਂ ਉੱਤੇ ਨਿਰਮਾਤਾ ਦੀ 1 ਸਾਲ ਦੀ ਵਾਰੰਟੀ ਹੁੰਦੀ ਹੈ। ਵੇਰਵਿਆਂ ਲਈ ਉਤਪਾਦ ਸੂਚੀ ਅਤੇ ਵਾਰੰਟੀ ਕਾਰਡ ਵੇਖੋ।",
        "general": "ਮੈਂ ਡਿਲਿਵਰੀ, ਵਾਪਸੀ ਅਤੇ ਰਿਫੰਡ, UPI ਜਾਂ ਕਾਰਡ ਭੁਗਤਾਨ ਅਤੇ ਉਤਪਾਦ ਵਾਰੰਟੀ ਬਾਰੇ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ। ਕੀ ਜਾਣਨਾ ਹੈ?",
    },
    "ur": {
        "greeting": "السلام علیکم! میں ترسیل، واپسی، ادائیگی اور وارنٹی میں مدد کر سکتا ہوں۔ آپ کیا جاننا چاہتے ہیں؟",
        "shipping": "₹999 سے زیادہ کے آرڈرز پر معیاری ترسیل مفت ہے۔ زیادہ تر آرڈرز 3–5 کاروباری دنوں میں پہنچتے ہیں۔ اپنے اکاؤنٹ میں آرڈر ٹریک کریں۔",
        "returns": "اہل اشیا کو ترسیل کے 7 دن کے اندر اصل حالت اور پیکنگ میں واپس یا تبدیل کیا جا سکتا ہے۔ مدد کے لیے سپورٹ ٹیم سے رابطہ کریں۔",
        "payments": "ہم UPI، کریڈٹ اور ڈیبٹ کارڈ، نیٹ بینکنگ اور دستیابی کے مطابق کیش آن ڈیلیوری قبول کرتے ہیں۔ ادائیگیاں محفوظ ہیں۔",
        "warranty": "زیادہ تر الیکٹرانکس پر کمپنی کی 1 سال کی وارنٹی ہوتی ہے۔ تفصیل کے لیے مصنوعات کی فہرست اور وارنٹی کارڈ دیکھیں۔",
        "general": "میں ترسیل، واپسی اور رقم کی واپسی، UPI یا کارڈ ادائیگی اور مصنوعات کی وارنٹی میں مدد کر سکتا ہوں۔ آپ کیا جاننا چاہتے ہیں؟",
    },
    "es": {
        "greeting": "¡Hola! Puedo ayudarte con envíos, devoluciones, pagos y garantías. ¿Qué te gustaría saber?",
        "shipping": "El envío estándar es gratis en pedidos superiores a ₹999. La mayoría llega en 3–5 días laborables. Puedes seguir tu pedido desde tu cuenta.",
        "returns": "Los artículos elegibles se pueden devolver o cambiar en un plazo de 7 días desde la entrega, en su estado y embalaje originales. Contacta con soporte para empezar.",
        "payments": "Aceptamos UPI, tarjetas de crédito y débito, banca electrónica y pago contra reembolso donde esté disponible. El pago se procesa de forma segura.",
        "warranty": "La mayoría de los productos electrónicos incluyen 1 año de garantía del fabricante. Consulta la ficha del producto y la tarjeta de garantía.",
        "general": "Puedo ayudarte con envíos, devoluciones y reembolsos, pagos con UPI o tarjeta y garantías. ¿Qué te gustaría saber?",
    },
    "fr": {
        "greeting": "Bonjour ! Je peux vous aider pour la livraison, les retours, les paiements et les garanties. Que souhaitez-vous savoir ?",
        "shipping": "La livraison standard est gratuite pour les commandes de plus de ₹999. La plupart arrivent sous 3 à 5 jours ouvrés. Suivez votre commande depuis votre compte.",
        "returns": "Les articles éligibles peuvent être retournés ou échangés dans les 7 jours suivant la livraison, dans leur état et emballage d’origine. Contactez l’assistance pour commencer.",
        "payments": "Nous acceptons UPI, les cartes de crédit et de débit, les virements bancaires et le paiement à la livraison lorsqu’il est disponible. Le paiement est sécurisé.",
        "warranty": "La plupart des appareils électroniques bénéficient d’une garantie fabricant d’un an. Consultez la fiche produit et la carte de garantie.",
        "general": "Je peux vous renseigner sur la livraison, les retours et remboursements, les paiements UPI ou par carte et les garanties. Que souhaitez-vous savoir ?",
    },
    "de": {
        "greeting": "Hallo! Ich helfe Ihnen gerne bei Versand, Rückgabe, Zahlung und Garantie. Was möchten Sie wissen?",
        "shipping": "Der Standardversand ist bei Bestellungen über ₹999 kostenlos. Die meisten Bestellungen treffen innerhalb von 3–5 Werktagen ein. Verfolgen Sie Ihre Bestellung in Ihrem Konto.",
        "returns": "Berechtigte Artikel können innerhalb von 7 Tagen nach Lieferung im Originalzustand und in der Originalverpackung zurückgegeben oder umgetauscht werden. Kontaktieren Sie den Support.",
        "payments": "Wir akzeptieren UPI, Kredit- und Debitkarten, Online-Banking und, sofern verfügbar, Nachnahme. Zahlungen werden sicher abgewickelt.",
        "warranty": "Die meisten Elektronikartikel haben eine einjährige Herstellergarantie. Einzelheiten finden Sie in der Produktbeschreibung und auf der Garantiekarte.",
        "general": "Ich helfe Ihnen gerne bei Versand, Rückgabe und Erstattung, UPI- oder Kartenzahlung sowie Produktgarantien. Was möchten Sie wissen?",
    },
    "ja": {
        "greeting": "こんにちは！配送、返品、お支払い、保証についてご案内します。何をお知りになりたいですか？",
        "shipping": "₹999を超えるご注文は通常配送が無料です。多くの商品は3〜5営業日で届きます。アカウントから注文を追跡できます。",
        "returns": "対象商品は、配送後7日以内であれば、元の状態と梱包で返品または交換できます。手続きはサポートへお問い合わせください。",
        "payments": "UPI、クレジットカード、デビットカード、ネットバンキング、および利用可能な地域では代金引換をご利用いただけます。お支払いは安全に処理されます。",
        "warranty": "ほとんどの電子機器にはメーカーの1年間保証が付いています。対象範囲は商品ページと保証書をご確認ください。",
        "general": "配送、返品・返金、UPIやカードでのお支払い、製品保証についてご案内します。何をお知りになりたいですか？",
    },
    "ar": {
        "greeting": "مرحباً! يمكنني مساعدتك بشأن الشحن والإرجاع والدفع والضمان. ماذا تريد أن تعرف؟",
        "shipping": "الشحن العادي مجاني للطلبات التي تزيد قيمتها على ₹999. تصل معظم الطلبات خلال 3 إلى 5 أيام عمل. يمكنك تتبع طلبك من حسابك.",
        "returns": "يمكن إرجاع المنتجات المؤهلة أو استبدالها خلال 7 أيام من التسليم، بحالتها وعبوتها الأصليتين. تواصل مع فريق الدعم للبدء.",
        "payments": "نقبل UPI وبطاقات الائتمان والخصم والخدمات المصرفية عبر الإنترنت والدفع عند الاستلام حيثما كان متاحاً. تتم معالجة المدفوعات بأمان.",
        "warranty": "تأتي معظم الأجهزة الإلكترونية مع ضمان من الشركة المصنعة لمدة سنة واحدة. راجع صفحة المنتج وبطاقة الضمان لمعرفة التفاصيل.",
        "general": "يمكنني مساعدتك بشأن الشحن والإرجاع واسترداد الأموال والدفع عبر UPI أو البطاقة وضمان المنتجات. ماذا تريد أن تعرف؟",
    },
}


class SoundFileAudioData(sr.AudioData):
    def get_flac_data(self, convert_rate=None, convert_width=None):
        """Encode 16-bit audio with libsndfile instead of SpeechRecognition's bundled FLAC binary."""
        wav_data = self.get_wav_data(convert_rate=convert_rate, convert_width=2)
        samples, sample_rate = sf.read(BytesIO(wav_data), dtype="int16")
        flac_data = BytesIO()
        sf.write(
            flac_data,
            samples,
            sample_rate,
            format="FLAC",
            subtype="PCM_16",
        )
        return flac_data.getvalue()


@st.cache_data
def get_recommender(products_df):
    products_df = products_df.reset_index(drop=True)
    descriptions = [
        f"{product['name']} {product['category']} {product['description']}"
        for product in products_df.to_dict("records")
    ]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(descriptions)
    return matrix


@st.cache_resource
def get_review_classifier():
    reviews = GENUINE_REVIEWS + SUSPICIOUS_REVIEWS
    labels = [0] * len(GENUINE_REVIEWS) + [1] * len(SUSPICIOUS_REVIEWS)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform(reviews)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(features, labels)
    return vectorizer, classifier


def answer_question(question, lang_code):
    query = question.casefold()
    language_keywords = INTENT_KEYWORDS.get(lang_code, INTENT_KEYWORDS["en"])
    responses = CHATBOT_RESPONSES.get(lang_code, CHATBOT_RESPONSES["en"])
    for intent in ("greeting", "payments", "warranty", "returns", "shipping"):
        if any(keyword.casefold() in query for keyword in language_keywords[intent]):
            return responses[intent]
    return responses["general"]


def speak(text, lang_code):
    audio = BytesIO()
    gTTS(text=text, lang=lang_code).write_to_fp(audio)
    audio.seek(0)
    return audio


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8f7ff 0%, #fff8f1 52%, #f0fbff 100%);
    }
    .hero {
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
        border-radius: 22px;
        color: white;
        background: linear-gradient(110deg, #5635c9, #9b42c7 56%, #f27675);
        box-shadow: 0 12px 32px rgba(86, 53, 201, 0.18);
    }
    .hero h1 { margin: 0 0 .35rem; font-size: 2.2rem; }
    .hero p { margin: 0; opacity: .92; font-size: 1.05rem; }
    .product-card {
        background: rgba(255,255,255,.92);
        border: 1px solid #ebe7f8;
        border-radius: 18px;
        padding: 1.1rem;
        margin-bottom: .8rem;
        box-shadow: 0 5px 18px rgba(50, 40, 100, .06);
    }
    .product-card h3 { margin: .25rem 0; color: #28223b; font-size: 1.05rem; }
    .muted { color: #706b7d; font-size: .92rem; }
    .price { color: #603dcc; font-weight: 700; font-size: 1.08rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🛍️ Smart E-Commerce AI &amp; Voice Assistant</h1>
      <p>Discover products, get quick answers, and shop with a little help from AI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

recommender_tab, chatbot_tab, reviews_tab = st.tabs(
    ["✨ Product Recommender", "🎙️ AI Voice & Text Chatbot", "🛡️ Fake Review Detector"]
)

with recommender_tab:
    st.subheader("Find your next favorite")
    st.write("Filter the catalog by category, choose a product, and explore three similar recommendations.")
    categories = ["All categories"] + sorted({item["category"] for item in PRODUCTS})
    selected_category = st.selectbox("Category filter", categories)
    products_df = pd.DataFrame(PRODUCTS)
    filtered_products_df = products_df.loc[
        (products_df["category"] == selected_category)
        if selected_category != "All categories"
        else slice(None)
    ].reset_index(drop=True)
    selected_name = st.selectbox(
        "Select a product",
        filtered_products_df["name"].tolist(),
    )
    selected_index = int(
        filtered_products_df.index[
            filtered_products_df["name"] == selected_name
        ][0]
    )
    product_matrix = get_recommender(filtered_products_df)
    similarity_scores = cosine_similarity(
        product_matrix[selected_index], product_matrix
    ).flatten()
    recommendations = [
        index
        for index in similarity_scores.argsort()[::-1]
        if index != selected_index
    ][:3]
    selected_product = filtered_products_df.iloc[selected_index]

    st.markdown(
        f"**Selected:** {selected_product['icon']} {selected_name} "
        f"— ₹{selected_product['price']:,}"
    )
    columns = st.columns(3)
    for column, index in zip(columns, recommendations):
        product = filtered_products_df.iloc[index]
        with column:
            st.markdown(
                f"""
                <div class="product-card">
                  <div style="font-size:2rem">{product['icon']}</div>
                  <h3>{product['name']}</h3>
                  <div class="muted">{product['category']}</div>
                  <p>{product['description']}</p>
                  <div class="price">₹{product['price']:,}</div>
                  <div class="muted">Match: {similarity_scores[index] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with chatbot_tab:
    st.subheader("Ask our shopping assistant")
    st.write("Choose your language, then type a question or record one. I can help with shipping, returns, payments, and warranty.")
    selected_language = st.selectbox("Response and speech language", list(LANGUAGES))
    speech_language = LANGUAGES[selected_language]["speech"]
    tts_language = LANGUAGES[selected_language]["tts"]
    audio_recording = st.audio_input("Record your question")
    if audio_recording is not None:
        if st.button("Transcribe and answer recording", key="voice_submit"):
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(BytesIO(audio_recording.getvalue())) as audio_source:
                    audio_data = recognizer.record(audio_source)
                audio_data = SoundFileAudioData(
                    audio_data.frame_data,
                    audio_data.sample_rate,
                    audio_data.sample_width,
                )
                question = recognizer.recognize_google(
                    audio_data,
                    language=speech_language,
                )
            except sr.UnknownValueError:
                st.error("Sorry, I couldn't understand that recording. Please try again or type your question.")
            except sr.RequestError as error:
                st.error(f"Speech recognition is unavailable right now: {error}")
            except (OSError, ValueError) as error:
                st.error(f"Could not read the recorded audio: {error}")
            else:
                st.markdown(f"**You said:** {question}")
                try:
                    response = answer_question(question, tts_language)
                    st.info(response)
                    st.audio(speak(response, tts_language), format="audio/mp3")
                except (gTTSError, OSError, ValueError) as error:
                    st.error(f"Could not generate spoken audio: {error}")

    with st.form("text_question_form"):
        text_question = st.text_input(
            "Or type your question",
            placeholder=f"Type in {selected_language} or any other language...",
        )
        submitted = st.form_submit_button("Get an answer")
    if submitted:
        if not text_question.strip():
            st.warning("Enter a question so I know how to help.")
        else:
            try:
                response = answer_question(text_question, tts_language)
                st.info(response)
                st.audio(speak(response, tts_language), format="audio/mp3")
            except (gTTSError, OSError, ValueError) as error:
                st.error(f"Could not generate spoken audio: {error}")

with reviews_tab:
    st.subheader("Check a product review")
    st.write("Paste a review to estimate whether it looks genuine or suspicious.")
    review_text = st.text_area(
        "Review text",
        height=150,
        placeholder="Paste the review you want to check...",
    )
    if st.button("Analyze review", type="primary"):
        if not review_text.strip():
            st.warning("Paste a review before analyzing it.")
        else:
            vectorizer, classifier = get_review_classifier()
            features = vectorizer.transform([review_text])
            probabilities = classifier.predict_proba(features)[0]
            predicted_label = classifier.predict(features)[0]
            confidence = probabilities[int(predicted_label)] * 100
            if predicted_label == 1:
                st.warning(f"🟠 Fake / Suspicious — confidence: {confidence:.1f}%")
                st.caption("This is an estimate from a small sample model, not a definitive judgment.")
            else:
                st.success(f"✅ Genuine — confidence: {confidence:.1f}%")
                st.caption("This is an estimate from a small sample model, not a definitive judgment.")
