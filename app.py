import re
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

COMMON_INTENT_KEYWORDS = {
    "delivery": (
        "delivery", "deliver", "shipping", "ship", "free delivery",
        "shipping time", "delivery time", "arrive", "arrival", "when will",
        "mysore", "mysuru", "ಮೈಸೂರು", "ಮೈಸೂರಿಗೆ", "मैसूर", "मैसूरु",
        "ಬೆಂಗಳೂರು", "ಬೆಂಗಳೂರುಗೆ", "ಬೆంగళూరు", "ಬೆಂಗಳೂರು", "बेंगलुरु",
        "बेंगलूरु", "बैंगलोर", "bengaluru", "bangalore", "hyderabad",
        "ಹೈದರಾಬಾದ್", "हैदराबाद", "chennai", "சென்னை", "चेन्नई",
        "mumbai", "मुंबई", "ಮುಂಬೈ", "delhi", "दिल्ली", "ದೆಹಲಿ",
        "pincode", "pin code", "postal code", "serviceable", "deliver to",
    ),
    "returns": (
        "return", "returns", "refund", "exchange", "replace", "replacement",
        "ರಿಟರ್ನ್", "ರಿಟರ್ನ್ ಮಾಡಿ", "ಹಿಂತಿರುಗಿ", "ಮರುಪಾವತಿ", "ಹಣ ವಾಪಸ್",
        "वापसी", "वापस", "रिटर्न", "रिफंड", "बदलना", "प्रतिस्थापन",
        "திரும்ப", "ரிட்டர்ன்", "பணத்தைத் திரும்ப", "பணத்தை திரும்ப",
        "மாற்ற", "తిరిగి", "రిటర్న్", "వాపసు", "రీఫండ్", "మార్చు",
        "return policy", "how do i return",
    ),
    "payments": (
        "payment", "payments", "pay", "upi", "gpay", "google pay", "phonepe",
        "phone pe", "paytm", "card", "credit card", "debit card", "cash on delivery",
        "cod", "net banking", "ಪೇಮೆಂಟ್", "ಪಾವತಿ", "ಯುಪಿಐ", "ಗೂಗಲ್ ಪೇ",
        "ಫೋನ್ ಪೇ", "ಕಾರ್ಡ್", "ಹಣ ಪಾವತಿ", "पेमेंट", "भुगतान",
        "यूपीआई", "गूगल पे", "फोनपे", "कार्ड", "नकद भुगतान", "பேமெண்ட்",
        "கட்டணம்", "யுபிஐ", "கூகுள் பே", "போன்பே", "கார்டு", "చెల్లింపు",
        "పేమెంట్", "యూపీఐ", "గూగుల్ పే", "ఫోన్ పే", "కార్డు",
    ),
    "tracking": (
        "track", "tracking", "order status", "where is my order", "my order",
        "order id", "awb", "consignment", "shipment status", "track order",
        "ಆರ್ಡರ್ ಟ್ರ್ಯಾಕ್", "ಆರ್ಡರ್ ಎಲ್ಲಿದೆ", "ಆರ್ಡರ್ ಸ್ಥಿತಿ", "ಆರ್ಡರ್ ಸ್ಥಿತಿ",
        "ಆರ್ಡರ್ ಎಲ್ಲಿ", "ऑर्डर ट्रैक", "ऑर्डर कहां", "ऑर्डर कहाँ",
        "ऑर्डर की स्थिति", "मेरा ऑर्डर", "ஆர்டர் எங்கே", "ஆர்டர் நிலை",
        "டிராக்", "ఆర్డర్ ఎక్కడ", "ఆర్డర్ స్థితి", "ట్రాక్",
    ),
    "offers": (
        "discount", "offer", "offers", "coupon", "promo code", "coupon code",
        "deal", "sale", "discount code", "ಡಿಸ್ಕೌಂಟ್", "ಆಫರ್", "ಕೂಪನ್",
        "ರಿಯಾಯಿತಿ", "ऑफर", "छूट", "डिस्काउंट", "कूपन", "सेल",
        "தள்ளுபடி", "ஆஃபர்", "கூப்பன்", "விலை குறைப்பு",
        "డిస్కౌంట్", "ఆఫర్", "కూపన్", "తగ్గింపు",
    ),
    "catalog": (
        "price", "prices", "cost", "catalog", "catalogue", "products", "product",
        "how much", "available products", "item", "electronics", "fashion",
        "apparel", "beauty", "personal care", "books", "home", "kitchen",
        "sports", "fitness", "ಬೆಲೆ", "ಬೆಲೆ ಎಷ್ಟು", "ದರ", "ಉತ್ಪನ್ನಗಳು",
        "ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್", "ಬಟ್ಟೆ", "ಸೌಂದರ್ಯ", "ಪುಸ್ತಕ", "ಮನೆ", "ಕ್ರೀಡೆ",
        "कीमत", "मूल्य", "दाम", "उत्पाद", "इलेक्ट्रॉनिक्स", "फैशन",
        "सौंदर्य", "किताब", "घर", "खेल", "कीमत कितनी",
        "விலை", "விலை என்ன", "பொருட்கள்", "எலக்ட்ரானிக்ஸ்", "ஆடை",
        "அழகு", "புத்தகம்", "வீடு", "விளையாட்டு",
        "ధర", "ధర ఎంత", "వస్తువులు", "ఎలక్ట్రానిక్స్", "ఫ్యాషన్",
        "అందం", "పుస్తకాలు", "ఇల్లు", "క్రీడలు",
    ),
    "warranty": (
        "warranty", "guarantee", "repair", "service center", "defective",
        "ಖಾತರಿ", "ವಾರಂಟಿ", "ದುರಸ್ತಿ", "ವಾರಂಟಿ ಅವಧಿ",
        "वारंटी", "गारंटी", "मरम्मत", "खराब",
        "உத்தரவாதம்", "வாரண்டி", "பழுது", "சேவை மையம்",
        "వారంటీ", "హామీ", "మరమ్మత్తు", "సర్వీస్ సెంటర్",
    ),
    "support": (
        "support", "customer care", "customer service", "contact", "help desk",
        "help", "helpline", "phone number", "email", "talk to someone",
        "ಗ್ರಾಹಕ ಸೇವೆ", "ಸಹಾಯ", "ಸಂಪರ್ಕ", "ಗ್ರಾಹಕ ಆರೈಕೆ",
        "ग्राहक सेवा", "सहायता", "संपर्क", "कस्टमर केयर",
        "வாடிக்கையாளர் சேவை", "உதவி", "தொடர்பு", "ஆதரவு",
        "కస్టమర్ కేర్", "సహాయం", "సంప్రదించండి", "కస్టమర్ సర్వీస్",
    ),
    "greeting": (
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "ನಮಸ್ಕಾರ", "ಹಲೋ",
        "ಶುಭೋದಯ", "ನಮಸ್ತೆ", "नमस्ते", "नमस्कार", "प्रणाम",
        "வணக்கம்", "நலமா", "హలో", "నమస్కారం", "శుభోదయం",
        "bonjour", "hola", "hallo", "こんにちは", "مرحبا",
    ),
    "thanks": (
        "thank you", "thanks", "thankyou", "ಧನ್ಯವಾದ", "ಧನ್ಯವಾದಗಳು",
        "शुक्रिया", "धन्यवाद", "நன்றி", "ధన్యవాదాలు", "ధన్యవాదములు",
        "gracias", "merci", "danke", "ありがとう", "شكرا", "شكرًا",
    ),
}

LOCAL_INTENT_KEYWORDS = {
    "en": {},
    "kn": {
        "delivery": ("ಡೆಲಿವರಿ", "ವಿತರಣೆ", "ಶಿಪ್ಪಿಂಗ್", "ತಲುಪುತ್ತದೆಯೇ", "ಎಷ್ಟು ದಿನ", "ಉಚಿತ ಡೆಲಿವರಿ", "delivery yavag", "mysore ge", "yavaga barutte"),
        "returns": ("ರಿಟರ್ನ್", "ರಿಟರ್ನ್ ಪಾಲಿಸಿ", "ಹಿಂತಿರುಗಿಸುವುದು", "ಹಣ ಹಿಂದಿರುಗಿಸಿ", "return madodu"),
        "payments": ("ಪೇಮೆಂಟ್", "ಯುಪಿಐ", "ಜಿಪೇ", "ಜೀ ಪೇ", "ಫೋನ್‌ಪೇ", "ಕ್ಯಾಶ್ ಆನ್ ಡೆಲಿವರಿ", "payment heg"),
        "tracking": ("ಆರ್ಡರ್ ಟ್ರ್ಯಾಕ್", "ಆರ್ಡರ್ ಎಲ್ಲಿ", "ಆರ್ಡರ್ ಸ್ಥಿತಿ", "ನನ್ನ ಆರ್ಡರ್", "order ellide"),
        "offers": ("ಡಿಸ್ಕೌಂಟ್", "ಆಫರ್", "ಕೂಪನ್", "ರಿಯಾಯಿತಿ", "discount ideya"),
        "catalog": ("ಬೆಲೆ", "ಬೆಲೆ ಎಷ್ಟು", "ದರ", "ಉತ್ಪನ್ನ", "ಪುಸ್ತಕ", "ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್", "ಫ್ಯಾಷನ್", "ಸೌಂದರ್ಯ", "ಮನೆ", "ಕ್ರೀಡೆ", "bele eshtu"),
        "warranty": ("ವಾರಂಟಿ", "ಖಾತರಿ", "ಬದಲಿ", "ದುರಸ್ತಿ"),
        "support": ("ಸಹಾಯ", "ಗ್ರಾಹಕ ಸೇವೆ", "ಸಂಪರ್ಕ", "ಫೋನ್ ಸಂಖ್ಯೆ"),
        "greeting": ("ನಮಸ್ಕಾರ", "ಹಲೋ", "ನಮಸ್ತೆ"),
        "thanks": ("ಧನ್ಯವಾದ", "ಧನ್ಯವಾದಗಳು"),
    },
    "hi": {
        "delivery": ("डिलीवरी", "डिलिवरी", "डिलीवरी कब", "कितने दिन", "मुफ्त डिलीवरी", "मैसूर", "मैसूरु", "बेंगलुरु", "delivery kab", "kitne din"),
        "returns": ("रिटर्न", "रिटर्न पॉलिसी", "वापसी", "रिफंड", "एक्सचेंज"),
        "payments": ("पेमेंट", "यूपीआई", "जीपे", "फोनपे", "कैश ऑन डिलीवरी"),
        "tracking": ("ऑर्डर ट्रैक", "मेरा ऑर्डर", "ऑर्डर कहां", "ऑर्डर की स्थिति"),
        "offers": ("डिस्काउंट", "ऑफर", "कूपन", "छूट", "प्रोमो कोड"),
        "catalog": ("कीमत", "कीमत कितनी", "मूल्य", "दाम", "उत्पाद", "इलेक्ट्रॉनिक्स", "फैशन", "सौंदर्य", "किताबें", "घर", "खेल", "keemat", "kitne ka"),
        "warranty": ("वारंटी", "गारंटी", "रिप्लेसमेंट", "मरम्मत"),
        "support": ("ग्राहक सेवा", "कस्टमर केयर", "सहायता", "संपर्क", "फोन नंबर"),
        "greeting": ("नमस्ते", "नमस्कार", "प्रणाम"),
        "thanks": ("धन्यवाद", "शुक्रिया"),
    },
    "ta": {
        "delivery": ("டெலிவரி", "விநியோகம்", "எத்தனை நாட்கள்", "இலவச டெலிவரி", "மைசூர்", "பெங்களூரு", "delivery eppo", "vilundhu"),
        "returns": ("ரிட்டர்ன்", "திரும்பப்பெறுதல்", "பணத்தைத் திரும்ப", "பரிமாற்றம்", "return seiya"),
        "payments": ("பேமெண்ட்", "யுபிஐ", "ஜிபே", "போன்பே", "கேஷ் ஆன் டெலிவரி", "payment eppadi"),
        "tracking": ("ஆர்டர் டிராக்", "என் ஆர்டர்", "ஆர்டர் எங்கே", "ஆர்டர் நிலை", "order enga"),
        "offers": ("டிஸ்கவுண்ட்", "தள்ளுபடி", "ஆஃபர்", "கூப்பன்", "discount irukka"),
        "catalog": ("விலை", "விலை என்ன", "பொருள்", "பொருட்கள்", "எலக்ட்ரானிக்ஸ்", "ஆடை", "அழகு", "புத்தகங்கள்", "வீடு", "விளையாட்டு", "vilai evvalavu"),
        "warranty": ("வாரண்டி", "உத்தரவாதம்", "மாற்றுப்பொருள்", "பழுது"),
        "support": ("வாடிக்கையாளர் சேவை", "தொடர்பு", "உதவி எண்", "ஆதரவு"),
        "greeting": ("வணக்கம்", "நலமா"),
        "thanks": ("நன்றி",),
    },
    "te": {
        "delivery": ("డెలివరీ", "రవాణా", "ఎన్ని రోజులు", "ఉచిత డెలివరీ", "మైసూరు", "బెంగళూరు", "delivery eppudu", "entha sepu"),
        "returns": ("రిటర్న్", "వాపసు", "రీఫండ్", "మార్పిడి", "return cheyali"),
        "payments": ("పేమెంట్", "యూపీఐ", "జిపే", "ఫోన్ పే", "క్యాష్ ఆన్ డెలివరీ", "payment ela"),
        "tracking": ("ఆర్డర్ ట్రాక్", "నా ఆర్డర్", "ఆర్డర్ ఎక్కడ", "ఆర్డర్ స్థితి", "order ekkada"),
        "offers": ("డిస్కౌంట్", "ఆఫర్", "కూపన్", "తగ్గింపు", "discount unda"),
        "catalog": ("ధర", "ధర ఎంత", "వస్తువు", "వస్తువులు", "ఎలక్ట్రానిక్స్", "ఫ్యాషన్", "అందం", "పుస్తకాలు", "ఇల్లు", "క్రీడలు", "dhara entha"),
        "warranty": ("వారంటీ", "హామీ", "రీప్లేస్‌మెంట్", "మరమ్మత్తు"),
        "support": ("కస్టమర్ కేర్", "సహాయం", "సంప్రదించండి", "ఫోన్ నంబర్"),
        "greeting": ("నమస్కారం", "హలో", "శుభోదయం"),
        "thanks": ("ధన్యవాదాలు", "ధన్యవాదములు"),
    },
}

LOCAL_INTENT_KEYWORDS.update({
    "ml": {
        "delivery": ("ഡെലിവറി", "എത്തും", "എത്താൻ എത്ര", "സൗജന്യ ഡെലിവറി"),
        "tracking": ("ഓർഡർ എവിടെ", "ഓർഡർ നില", "എന്റെ ഓർഡർ"),
        "offers": ("ഡിസ്കൗണ്ട്", "ഓഫർ", "കൂപ്പൺ", "വിലക്കുറവ്"),
        "catalog": ("വില", "വില എത്ര", "പുസ്തകം", "വസ്ത്രം", "സൗന്ദര്യം", "വീട്", "കായികം"),
        "support": ("ഉപഭോക്തൃ സേവനം", "സഹായം", "ബന്ധപ്പെടുക"),
        "thanks": ("നന്ദി",),
    },
    "mr": {
        "delivery": ("डिलिव्हरी", "पोहोचेल", "मोफत डिलिव्हरी", "म्हैसूर", "बेंगळुरू"),
        "tracking": ("ऑर्डर कुठे", "ऑर्डरची स्थिती", "माझी ऑर्डर"),
        "offers": ("डिस्काउंट", "ऑफर", "कूपन", "सवलत"),
        "catalog": ("किंमत", "किंमत किती", "पुस्तके", "फॅशन", "सौंदर्य", "घर", "क्रीडा"),
        "support": ("ग्राहक सेवा", "मदत", "संपर्क"),
        "thanks": ("धन्यवाद",),
    },
    "bn": {
        "delivery": ("ডেলিভারি", "কবে পৌঁছাবে", "বিনামূল্যে ডেলিভারি"),
        "tracking": ("অর্ডার কোথায়", "অর্ডারের অবস্থা", "আমার অর্ডার"),
        "offers": ("ডিসকাউন্ট", "অফার", "কুপন", "ছাড়"),
        "catalog": ("দাম", "দাম কত", "বই", "ফ্যাশন", "সৌন্দর্য", "বাড়ি", "খেলাধুলা"),
        "support": ("গ্রাহক পরিষেবা", "সহায়তা", "যোগাযোগ"),
        "thanks": ("ধন্যবাদ",),
    },
    "gu": {
        "delivery": ("ડિલિવરી", "ક્યારે પહોંચશે", "મફત ડિલિવરી"),
        "returns": ("રિટર્ન", "રિફંડ", "બદલી", "પાછા આપવું"),
        "payments": ("પેમેન્ટ", "ચુકવણી", "યુપીઆઈ", "ફોનપે", "કાર્ડ", "કેશ ઓન ડિલિવરી"),
        "tracking": ("ઓર્ડર ક્યાં", "ઓર્ડરની સ્થિતિ", "મારો ઓર્ડર"),
        "offers": ("ડિસ્કાઉન્ટ", "ઓફર", "કૂપન", "છૂટ"),
        "catalog": ("કિંમત", "કિંમત કેટલી", "પુસ્તકો", "ફેશન", "સૌંદર્ય", "ઘર", "રમતગમત"),
        "support": ("ગ્રાહક સેવા", "મદદ", "સંપર્ક"),
        "thanks": ("આભાર",),
    },
    "pa": {
        "delivery": ("ਡਿਲਿਵਰੀ", "ਕਦੋਂ ਪਹੁੰਚੇਗਾ", "ਮੁਫ਼ਤ ਡਿਲਿਵਰੀ"),
        "tracking": ("ਆਰਡਰ ਕਿੱਥੇ", "ਆਰਡਰ ਦੀ ਸਥਿਤੀ", "ਮੇਰਾ ਆਰਡਰ"),
        "offers": ("ਡਿਸਕਾਊਂਟ", "ਆਫ਼ਰ", "ਕੂਪਨ", "ਛੋਟ"),
        "catalog": ("ਕੀਮਤ", "ਕੀਮਤ ਕਿੰਨੀ", "ਕਿਤਾਬਾਂ", "ਫੈਸ਼ਨ", "ਸੁੰਦਰਤਾ", "ਘਰ", "ਖੇਡਾਂ"),
        "support": ("ਗਾਹਕ ਸੇਵਾ", "ਮਦਦ", "ਸੰਪਰਕ"),
        "thanks": ("ਧੰਨਵਾਦ",),
    },
    "ur": {
        "delivery": ("ڈیلیوری", "کب پہنچے گا", "مفت ترسیل"),
        "tracking": ("آرڈر کہاں", "آرڈر کی حالت", "میرا آرڈر"),
        "offers": ("ڈسکاؤنٹ", "آفر", "کوپن", "رعایت"),
        "catalog": ("قیمت", "کتنی قیمت", "کتابیں", "فیشن", "خوبصورتی", "گھر", "کھیل"),
        "support": ("کسٹمر سروس", "مدد", "رابطہ"),
        "thanks": ("شکریہ",),
    },
    "es": {
        "delivery": ("envío", "entrega", "cuánto tarda", "gratis", "mysore", "bengaluru"),
        "returns": ("devolución", "reembolso", "cambio"),
        "payments": ("pago", "tarjeta", "contra reembolso"),
        "tracking": ("seguimiento", "estado del pedido", "dónde está mi pedido"),
        "offers": ("descuento", "oferta", "cupón", "código promocional"),
        "catalog": ("precio", "cuánto cuesta", "libros", "ropa", "belleza", "cocina", "deporte"),
        "warranty": ("garantía", "reparación", "sustitución"),
        "support": ("atención al cliente", "contacto", "ayuda"),
        "greeting": ("buenos días", "buenas tardes"),
        "thanks": ("gracias", "muchas gracias"),
    },
    "fr": {
        "delivery": ("livraison", "expédition", "délai", "gratuite", "mysore", "bengaluru"),
        "returns": ("retour", "remboursement", "échange"),
        "payments": ("paiement", "carte", "contre remboursement"),
        "tracking": ("suivi", "statut de commande", "où est ma commande"),
        "offers": ("réduction", "offre", "coupon", "code promo"),
        "catalog": ("prix", "combien coûte", "livres", "vêtements", "beauté", "cuisine", "sport"),
        "warranty": ("garantie", "réparation", "remplacement"),
        "support": ("service client", "contact", "aide"),
        "thanks": ("merci", "je vous remercie"),
    },
    "de": {
        "delivery": ("versand", "lieferung", "lieferzeit", "kostenlos", "mysore", "bengaluru"),
        "returns": ("rückgabe", "erstattung", "umtausch"),
        "payments": ("zahlung", "karte", "nachnahme"),
        "tracking": ("sendungsverfolgung", "bestellstatus", "wo ist meine bestellung"),
        "offers": ("rabatt", "angebot", "gutschein", "coupon"),
        "catalog": ("preis", "wie viel kostet", "bücher", "kleidung", "beauty", "küche", "sport"),
        "warranty": ("garantie", "reparatur", "ersatz"),
        "support": ("kundenservice", "kontakt", "hilfe"),
        "thanks": ("danke", "vielen dank"),
    },
    "ja": {
        "delivery": ("配送", "配達", "何日", "送料無料", "マイソール", "バンガロール"),
        "returns": ("返品", "返金", "交換"),
        "payments": ("支払い", "カード", "代金引換"),
        "tracking": ("注文状況", "追跡", "注文はどこ"),
        "offers": ("割引", "セール", "クーポン"),
        "catalog": ("価格", "いくら", "本", "服", "美容", "キッチン", "スポーツ"),
        "warranty": ("保証", "修理", "交換"),
        "support": ("カスタマーサポート", "問い合わせ", "連絡"),
        "thanks": ("ありがとう", "ありがとうございます"),
    },
    "ar": {
        "delivery": ("الشحن", "التوصيل", "كم يستغرق", "توصيل مجاني", "ميسور", "بنغالور"),
        "returns": ("إرجاع", "استرداد", "استبدال"),
        "payments": ("الدفع", "بطاقة", "الدفع عند الاستلام"),
        "tracking": ("تتبع الطلب", "حالة الطلب", "أين طلبي"),
        "offers": ("خصم", "عرض", "قسيمة", "كوبون"),
        "catalog": ("السعر", "كم السعر", "كتب", "ملابس", "جمال", "مطبخ", "رياضة"),
        "warranty": ("الضمان", "إصلاح", "استبدال"),
        "support": ("خدمة العملاء", "الدعم", "تواصل"),
        "thanks": ("شكرا", "شكرًا", "شكراً"),
    },
})

DETAIL_RESPONSES = {
    "en": {
        "location": "We currently ship within India. Standard delivery usually takes 3–5 business days and is free on orders over ₹999. I can't verify a specific address here; enter your PIN code at checkout to confirm serviceability.",
        "tracking": "I can't access live order records here. Sign in and open Account → My Orders to see the latest status and tracking link. If tracking has not updated, contact support with your order ID.",
        "offers": "I don't have live coupon or sale data, so I won't guess at a discount. Check the Offers section or apply a coupon at checkout to see current eligible deals.",
        "catalog": "Our catalog includes Electronics, Fashion & Apparel, Beauty & Personal Care, Books, Home & Kitchen, and Sports & Fitness.",
        "support": "For order-specific help, open Account → Help / Contact Support and include your order ID. I don't have a verified phone number or email address to share.",
        "thanks": "You're welcome! Ask me about delivery, returns, payments, order tracking, offers, products and prices, warranty, or customer support.",
        "fallback": "I understood your question as “{question}”, but couldn't match it to an available topic. You can ask about delivery/location, returns/refunds, payment methods, order tracking, discounts, product prices/catalog, warranty/replacement, or customer support.",
        "product_price": "{name} is listed at ₹{price:,}. Prices can change, so the product page has the latest price.",
        "category_products": "{category} items in our catalog: {items}.",
    },
    "kn": {
        "location": "ನಾವು ಭಾರತದೆಲ್ಲೆಡೆ ವಿತರಣೆ ಮಾಡುತ್ತೇವೆ. ಸಾಮಾನ್ಯವಾಗಿ 3–5 ಕೆಲಸದ ದಿನಗಳು ಬೇಕಾಗುತ್ತವೆ; ₹999 ಕ್ಕಿಂತ ಹೆಚ್ಚಿನ ಆರ್ಡರ್‌ಗಳಿಗೆ ಉಚಿತ ವಿತರಣೆ. ನಿರ್ದಿಷ್ಟ ವಿಳಾಸದ ಸೇವೆಯನ್ನು ಇಲ್ಲಿ ಪರಿಶೀಲಿಸಲಾಗುವುದಿಲ್ಲ—ಚೆಕ್‌ಔಟ್‌ನಲ್ಲಿ ಪಿನ್ ಕೋಡ್ ನಮೂದಿಸಿ.",
        "tracking": "ಇಲ್ಲಿ ನೇರ ಆರ್ಡರ್ ಸ್ಥಿತಿಯನ್ನು ನೋಡಲು ಸಾಧ್ಯವಿಲ್ಲ. ಖಾತೆಗೆ ಸೈನ್ ಇನ್ ಮಾಡಿ → ನನ್ನ ಆರ್ಡರ್‌ಗಳು ತೆರೆಯಿರಿ; ಅಲ್ಲಿ ಸ್ಥಿತಿ ಮತ್ತು ಟ್ರ್ಯಾಕಿಂಗ್ ಲಿಂಕ್ ಸಿಗುತ್ತದೆ. ಆರ್ಡರ್ ಐಡಿಯೊಂದಿಗೆ ಬೆಂಬಲವನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "offers": "ಪ್ರಸ್ತುತ ಕೂಪನ್ ಅಥವಾ ಮಾರಾಟದ ಮಾಹಿತಿಗೆ ನನಗೆ ಪ್ರವೇಶವಿಲ್ಲ. ಈಗಿನ ಆಫರ್‌ಗಳಿಗಾಗಿ Offers ವಿಭಾಗ ನೋಡಿ ಅಥವಾ ಚೆಕ್‌ಔಟ್‌ನಲ್ಲಿ ಕೂಪನ್ ನಮೂದಿಸಿ.",
        "catalog": "ನಮ್ಮಲ್ಲಿ ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್, ಫ್ಯಾಷನ್ ಮತ್ತು ಉಡುಪು, ಸೌಂದರ್ಯ ಮತ್ತು ವೈಯಕ್ತಿಕ ಆರೈಕೆ, ಪುಸ್ತಕಗಳು, ಮನೆ ಮತ್ತು ಅಡುಗೆಮನೆ, ಕ್ರೀಡೆ ಮತ್ತು ಫಿಟ್‌ನೆಸ್ ವಿಭಾಗಗಳಿವೆ.",
        "support": "ಆರ್ಡರ್ ಸಹಾಯಕ್ಕಾಗಿ ಖಾತೆ → ಸಹಾಯ / ಗ್ರಾಹಕ ಸೇವೆ ತೆರೆಯಿರಿ ಮತ್ತು ಆರ್ಡರ್ ಐಡಿ ನೀಡಿ. ಪರಿಶೀಲಿತ ಫೋನ್ ಸಂಖ್ಯೆ ಅಥವಾ ಇಮೇಲ್ ನನ್ನ ಬಳಿ ಇಲ್ಲ.",
        "thanks": "ಸ್ವಾಗತ! ವಿತರಣೆ, ವಾಪಸಿ, ಪಾವತಿ, ಆರ್ಡರ್ ಟ್ರ್ಯಾಕಿಂಗ್, ಆಫರ್, ಉತ್ಪನ್ನದ ಬೆಲೆ, ವಾರಂಟಿ ಅಥವಾ ಗ್ರಾಹಕ ಸೇವೆ ಬಗ್ಗೆ ಕೇಳಿ.",
        "fallback": "ನಿಮ್ಮ ಪ್ರಶ್ನೆ “{question}” ಎಂದು ಅರ್ಥವಾಯಿತು, ಆದರೆ ಲಭ್ಯವಿರುವ ವಿಷಯಕ್ಕೆ ಹೊಂದಿಸಲು ಆಗಲಿಲ್ಲ. ವಿತರಣೆ/ಸ್ಥಳ, ವಾಪಸಿ/ರಿಫಂಡ್, ಪಾವತಿ, ಆರ್ಡರ್ ಟ್ರ್ಯಾಕಿಂಗ್, ರಿಯಾಯಿತಿ, ಉತ್ಪನ್ನದ ಬೆಲೆ, ವಾರಂಟಿ ಅಥವಾ ಗ್ರಾಹಕ ಸೇವೆ ಬಗ್ಗೆ ಕೇಳಿ.",
        "product_price": "{name} ಬೆಲೆ ₹{price:,}. ಬೆಲೆ ಬದಲಾಗಬಹುದು; ಇತ್ತೀಚಿನ ಬೆಲೆಗೆ ಉತ್ಪನ್ನ ಪುಟ ನೋಡಿ.",
        "category_products": "ನಮ್ಮ {category} ವಿಭಾಗದಲ್ಲಿರುವ ವಸ್ತುಗಳು: {items}.",
    },
    "hi": {
        "location": "हम पूरे भारत में डिलीवरी करते हैं। आम तौर पर 3–5 कार्यदिवस लगते हैं; ₹999 से अधिक के ऑर्डर पर डिलीवरी मुफ़्त है। किसी पते की उपलब्धता यहाँ जाँच नहीं सकता—चेकआउट पर पिन कोड डालें।",
        "tracking": "मैं यहाँ लाइव ऑर्डर रिकॉर्ड नहीं देख सकता। खाते में साइन इन करके Account → My Orders खोलें; वहाँ स्थिति और ट्रैकिंग लिंक मिलेगा। अपडेट न हो तो ऑर्डर ID के साथ सहायता से संपर्क करें।",
        "offers": "मेरे पास लाइव कूपन या सेल की जानकारी नहीं है, इसलिए छूट का अनुमान नहीं दूँगा। मौजूदा ऑफ़र के लिए Offers सेक्शन देखें या चेकआउट पर कूपन लगाएँ।",
        "catalog": "हमारे कैटलॉग में Electronics, Fashion & Apparel, Beauty & Personal Care, Books, Home & Kitchen और Sports & Fitness श्रेणियाँ हैं।",
        "support": "ऑर्डर से जुड़ी मदद के लिए Account → Help / Contact Support खोलें और ऑर्डर ID दें। मेरे पास सत्यापित फोन नंबर या ईमेल नहीं है।",
        "thanks": "आपका स्वागत है! डिलीवरी/स्थान, रिटर्न/रिफंड, भुगतान, ऑर्डर ट्रैकिंग, ऑफ़र, उत्पाद की कीमत, वारंटी या ग्राहक सेवा पूछें।",
        "fallback": "आपका सवाल “{question}” समझा, लेकिन उसे उपलब्ध विषय से नहीं जोड़ पाया। डिलीवरी/स्थान, रिटर्न/रिफंड, भुगतान, ऑर्डर ट्रैकिंग, छूट, उत्पाद की कीमत, वारंटी या ग्राहक सेवा पूछें।",
        "product_price": "{name} की कीमत ₹{price:,} है। कीमत बदल सकती है; ताज़ा कीमत उत्पाद पेज पर देखें।",
        "category_products": "{category} में उपलब्ध उत्पाद: {items}।",
    },
    "ta": {
        "location": "இந்தியாவிற்குள் விநியோகம் செய்கிறோம். வழக்கமாக 3–5 வேலை நாட்கள் ஆகும்; ₹999-க்கு மேற்பட்ட ஆர்டர்களுக்கு இலவசம். குறிப்பிட்ட முகவரியை இங்கே சரிபார்க்க முடியாது—செக் அவுட்டில் பின் குறியீட்டை உள்ளிடவும்.",
        "tracking": "இங்கே நேரடி ஆர்டர் விவரங்களைப் பார்க்க முடியாது. கணக்கில் உள்நுழைந்து Account → My Orders என்பதைத் திறக்கவும்; நிலை மற்றும் கண்காணிப்பு இணைப்பு அங்கே இருக்கும். ஆர்டர் ID-யுடன் ஆதரவைத் தொடர்புகொள்ளவும்.",
        "offers": "நேரடி கூப்பன் அல்லது விற்பனைத் தகவல் என்னிடம் இல்லை. தற்போதைய சலுகைகளுக்கு Offers பகுதியைப் பார்க்கவும் அல்லது செக் அவுட்டில் கூப்பனை உள்ளிடவும்.",
        "catalog": "எங்கள் பட்டியலில் Electronics, Fashion & Apparel, Beauty & Personal Care, Books, Home & Kitchen, Sports & Fitness ஆகிய பிரிவுகள் உள்ளன.",
        "support": "ஆர்டர் உதவிக்கு Account → Help / Contact Support என்பதைத் திறந்து ஆர்டர் ID-ஐ வழங்கவும். சரிபார்க்கப்பட்ட தொலைபேசி அல்லது மின்னஞ்சல் என்னிடம் இல்லை.",
        "thanks": "நன்றி! விநியோகம்/இடம், திரும்பப்பெறுதல், கட்டணம், ஆர்டர் கண்காணிப்பு, சலுகைகள், பொருள் விலை, உத்தரவாதம் அல்லது வாடிக்கையாளர் சேவை பற்றி கேளுங்கள்.",
        "fallback": "உங்கள் கேள்வி “{question}” என்று புரிந்தது; ஆனால் கிடைக்கும் தலைப்புடன் பொருத்த முடியவில்லை. விநியோகம்/இடம், திரும்பப்பெறுதல், கட்டணம், ஆர்டர் கண்காணிப்பு, தள்ளுபடி, பொருள் விலை, உத்தரவாதம் அல்லது ஆதரவு பற்றி கேளுங்கள்.",
        "product_price": "{name} விலை ₹{price:,}. விலை மாறலாம்; சமீபத்திய விலைக்கு பொருள் பக்கத்தைப் பார்க்கவும்.",
        "category_products": "எங்கள் {category} பிரிவில் உள்ள பொருட்கள்: {items}.",
    },
    "te": {
        "location": "మేము భారతదేశంలో డెలివరీ చేస్తాము. సాధారణంగా 3–5 పని దినాలు పడతాయి; ₹999 కంటే ఎక్కువ ఆర్డర్లకు ఉచితం. నిర్దిష్ట చిరునామా సేవను ఇక్కడ నిర్ధారించలేను—చెక్‌అవుట్‌లో పిన్ కోడ్ నమోదు చేయండి.",
        "tracking": "ఇక్కడ ప్రత్యక్ష ఆర్డర్ వివరాలు చూడలేను. ఖాతాలో సైన్ ఇన్ చేసి Account → My Orders తెరవండి; స్థితి, ట్రాకింగ్ లింక్ అక్కడ ఉంటాయి. ఆర్డర్ IDతో సపోర్ట్‌ను సంప్రదించండి.",
        "offers": "ప్రస్తుత కూపన్ లేదా సేల్ సమాచారం నాకు అందుబాటులో లేదు. తాజా ఆఫర్ల కోసం Offers విభాగం చూడండి లేదా చెక్అవుట్‌లో కూపన్ నమోదు చేయండి.",
        "catalog": "మా కేటలాగ్‌లో Electronics, Fashion & Apparel, Beauty & Personal Care, Books, Home & Kitchen, Sports & Fitness విభాగాలు ఉన్నాయి.",
        "support": "ఆర్డర్ సహాయం కోసం Account → Help / Contact Support తెరిచి ఆర్డర్ ID ఇవ్వండి. ధృవీకరించిన ఫోన్ నంబర్ లేదా ఇమెయిల్ నా వద్ద లేదు.",
        "thanks": "స్వాగతం! డెలివరీ/ప్రాంతం, రిటర్న్/రీఫండ్, చెల్లింపులు, ఆర్డర్ ట్రాకింగ్, ఆఫర్లు, ఉత్పత్తి ధర, వారంటీ లేదా కస్టమర్ సపోర్ట్ గురించి అడగండి.",
        "fallback": "మీ ప్రశ్న “{question}” అని అర్థమైంది, కానీ అందుబాటులో ఉన్న అంశంతో సరిపోలలేదు. డెలివరీ/ప్రాంతం, రిటర్న్/రీఫండ్, చెల్లింపులు, ఆర్డర్ ట్రాకింగ్, డిస్కౌంట్, ఉత్పత్తి ధర, వారంటీ లేదా సపోర్ట్ గురించి అడగండి.",
        "product_price": "{name} ధర ₹{price:,}. ధర మారవచ్చు; తాజా ధరకు ఉత్పత్తి పేజీ చూడండి.",
        "category_products": "మా {category} విభాగంలోని ఉత్పత్తులు: {items}.",
    },
}

DETAIL_RESPONSES.update({
    "ml": {
        "location": "ഞങ്ങൾ ഇന്ത്യയിലുടനീളം ഡെലിവറി നടത്തുന്നു. സാധാരണയായി 3–5 പ്രവൃത്തി ദിവസങ്ങൾ; ₹999-ന് മുകളിലുള്ള ഓർഡറുകൾക്ക് സൗജന്യം. വിലാസം ഇവിടെ പരിശോധിക്കാനാവില്ല—ചെക്ക്ഔട്ടിൽ പിൻകോഡ് നൽകുക.",
        "tracking": "ഇവിടെ തത്സമയ ഓർഡർ വിവരങ്ങൾ കാണാനാകില്ല. അക്കൗണ്ടിൽ സൈൻ ഇൻ ചെയ്ത് Account → My Orders തുറക്കുക. ഓർഡർ ഐഡിയുമായി പിന്തുണയെ ബന്ധപ്പെടുക.",
        "offers": "നിലവിലെ കൂപ്പൺ വിവരങ്ങൾ എനിക്ക് ലഭ്യമല്ല. പുതിയ ഓഫറുകൾക്കായി Offers വിഭാഗം നോക്കുക അല്ലെങ്കിൽ ചെക്ക്ഔട്ടിൽ കൂപ്പൺ നൽകുക.",
        "catalog": "ഞങ്ങളുടെ കാറ്റലോഗിൽ Electronics, Fashion & Apparel, Beauty & Personal Care, Books, Home & Kitchen, Sports & Fitness വിഭാഗങ്ങളുണ്ട്.",
        "support": "ഓർഡർ സഹായത്തിന് Account → Help / Contact Support തുറന്ന് ഓർഡർ ഐഡി നൽകുക. സ്ഥിരീകരിച്ച ഫോൺ നമ്പറോ ഇമെയിലോ എനിക്ക് ലഭ്യമല്ല.",
        "thanks": "സ്വാഗതം! ഡെലിവറി, മടക്കം, പേയ്മെന്റ്, ഓർഡർ ട്രാക്കിംഗ്, ഓഫറുകൾ, വില, വാറന്റി, പിന്തുണ എന്നിവ ചോദിക്കാം.",
        "fallback": "നിങ്ങളുടെ ചോദ്യം “{question}” ആണെന്ന് മനസ്സിലാക്കി; പക്ഷേ വിഷയം തിരിച്ചറിയാനായില്ല. ഡെലിവറി/സ്ഥലം, മടക്കം, പേയ്മെന്റ്, ഓർഡർ ട്രാക്കിംഗ്, ഓഫർ, വില, വാറന്റി അല്ലെങ്കിൽ പിന്തുണ ചോദിക്കൂ.",
        "product_price": "{name} വില ₹{price:,}. വില മാറാം; പുതുക്കിയ വിലയ്ക്ക് ഉൽപ്പന്ന പേജ് നോക്കുക.",
        "category_products": "{category} വിഭാഗത്തിലെ ഉൽപ്പന്നങ്ങൾ: {items}.",
    },
    "mr": {
        "location": "आम्ही भारतात डिलिव्हरी करतो. साधारण 3–5 कामकाजाचे दिवस लागतात; ₹999 पेक्षा जास्त ऑर्डरवर मोफत. पत्ता येथे तपासता येत नाही—चेकआउटमध्ये पिन कोड द्या.",
        "tracking": "येथे लाइव्ह ऑर्डर माहिती पाहता येत नाही. खात्यात साइन इन करून Account → My Orders उघडा. ऑर्डर ID सह सहाय्याशी संपर्क साधा.",
        "offers": "सध्याच्या कूपनची थेट माहिती माझ्याकडे नाही. Offers विभाग पाहा किंवा चेकआउटमध्ये कूपन वापरा.",
        "catalog": "आमच्या कॅटलॉगमध्ये Electronics, Fashion & Apparel, Beauty & Personal Care, Books, Home & Kitchen आणि Sports & Fitness आहेत.",
        "support": "ऑर्डर मदतीसाठी Account → Help / Contact Support उघडा आणि ऑर्डर ID द्या. सत्यापित फोन किंवा ईमेल माझ्याकडे नाही.",
        "thanks": "स्वागत आहे! डिलिव्हरी, परतावा, पेमेंट, ऑर्डर ट्रॅकिंग, ऑफर, किंमत, वॉरंटी किंवा ग्राहक सेवेसाठी विचारा.",
        "fallback": "तुमचा प्रश्न “{question}” समजला, पण विषय ओळखता आला नाही. डिलिव्हरी/ठिकाण, परतावा, पेमेंट, ऑर्डर ट्रॅकिंग, सूट, किंमत, वॉरंटी किंवा सहाय्याबद्दल विचारा.",
        "product_price": "{name} ची किंमत ₹{price:,} आहे. किंमत बदलू शकते; अद्ययावत किंमतीसाठी उत्पादनाचे पान पाहा.",
        "category_products": "{category} मधील उत्पादने: {items}.",
    },
    "bn": {
        "location": "আমরা ভারতের মধ্যে ডেলিভারি করি। সাধারণত 3–5 কর্মদিবস লাগে; ₹999-এর বেশি অর্ডারে বিনামূল্যে। ঠিকানা এখানে যাচাই করা যায় না—চেকআউটে পিন কোড দিন।",
        "tracking": "এখানে লাইভ অর্ডার দেখা যায় না। অ্যাকাউন্টে সাইন ইন করে Account → My Orders খুলুন। অর্ডার ID দিয়ে সহায়তায় যোগাযোগ করুন।",
        "offers": "বর্তমান কুপনের তথ্য আমার কাছে নেই। Offers বিভাগ দেখুন বা চেকআউটে কুপন প্রয়োগ করুন।",
        "catalog": "আমাদের ক্যাটালগে Electronics, Fashion & Apparel, Beauty & Personal Care, Books, Home & Kitchen এবং Sports & Fitness রয়েছে।",
        "support": "অর্ডার সহায়তার জন্য Account → Help / Contact Support খুলে অর্ডার ID দিন। যাচাইকৃত ফোন বা ইমেল আমার কাছে নেই।",
        "thanks": "স্বাগতম! ডেলিভারি, ফেরত, পেমেন্ট, অর্ডার ট্র্যাকিং, অফার, পণ্যের দাম, ওয়ারেন্টি বা সহায়তা সম্পর্কে জিজ্ঞাসা করুন।",
        "fallback": "আপনার প্রশ্ন “{question}” বুঝেছি, তবে বিষয়টি শনাক্ত করতে পারিনি। ডেলিভারি/স্থান, ফেরত, পেমেন্ট, অর্ডার ট্র্যাকিং, ছাড়, পণ্যের দাম, ওয়ারেন্টি বা সহায়তা সম্পর্কে জিজ্ঞাসা করুন।",
        "product_price": "{name}-এর দাম ₹{price:,}। দাম বদলাতে পারে; সর্বশেষ দাম পণ্যের পাতায় দেখুন।",
        "category_products": "{category} বিভাগে আমাদের পণ্য: {items}।",
    },
    "gu": {
        "location": "અમે ભારતમાં ડિલિવરી કરીએ છીએ. સામાન્ય રીતે 3–5 કામકાજના દિવસો લાગે છે; ₹999થી વધુના ઓર્ડર પર મફત. અહીં સરનામું ચકાસી શકાતું નથી—ચેકઆઉટમાં પિન કોડ દાખલ કરો.",
        "tracking": "અહીં લાઇવ ઓર્ડર જોઈ શકાતો નથી. ખાતામાં સાઇન ઇન કરીને Account → My Orders ખોલો. ઓર્ડર ID સાથે સહાયનો સંપર્ક કરો.",
        "offers": "હાલના કૂપનની માહિતી મારી પાસે નથી. Offers વિભાગ જુઓ અથવા ચેકઆઉટમાં કૂપન લગાવો.",
        "catalog": "અમારા કેટલોગમાં Electronics, Fashion & Apparel, Beauty & Personal Care, Books, Home & Kitchen અને Sports & Fitness છે.",
        "support": "ઓર્ડરની મદદ માટે Account → Help / Contact Support ખોલીને ઓર્ડર ID આપો. ચકાસેલો ફોન અથવા ઇમેલ મારી પાસે નથી.",
        "thanks": "આભાર! ડિલિવરી, પરત, ચુકવણી, ઓર્ડર ટ્રેકિંગ, ઓફર, કિંમત, વોરંટી અથવા ગ્રાહક સેવા વિશે પૂછો.",
        "fallback": "તમારો પ્રશ્ન “{question}” સમજ્યો, પરંતુ વિષય ઓળખી શક્યો નહીં. ડિલિવરી/સ્થળ, પરત, ચુકવણી, ઓર્ડર ટ્રેકિંગ, છૂટ, કિંમત, વોરંટી અથવા સહાય વિશે પૂછો.",
        "product_price": "{name} ની કિંમત ₹{price:,} છે. કિંમત બદલાઈ શકે છે; નવીનતમ કિંમત માટે ઉત્પાદનનું પાનું જુઓ.",
        "category_products": "{category} માં ઉપલબ્ધ વસ્તુઓ: {items}.",
    },
    "pa": {
        "location": "ਅਸੀਂ ਭਾਰਤ ਵਿੱਚ ਡਿਲਿਵਰੀ ਕਰਦੇ ਹਾਂ। ਆਮ ਤੌਰ 'ਤੇ 3–5 ਕੰਮਕਾਜੀ ਦਿਨ ਲੱਗਦੇ ਹਨ; ₹999 ਤੋਂ ਵੱਧ ਦੇ ਆਰਡਰ ਮੁਫ਼ਤ ਹਨ। ਪਤਾ ਇੱਥੇ ਜਾਂਚਿਆ ਨਹੀਂ ਜਾ ਸਕਦਾ—ਚੈਕਆਉਟ 'ਤੇ ਪਿਨ ਕੋਡ ਦਿਓ।",
        "tracking": "ਇੱਥੇ ਲਾਈਵ ਆਰਡਰ ਨਹੀਂ ਵੇਖ ਸਕਦਾ। ਖਾਤੇ ਵਿੱਚ ਸਾਈਨ ਇਨ ਕਰਕੇ Account → My Orders ਖੋਲ੍ਹੋ। ਆਰਡਰ ID ਨਾਲ ਸਹਾਇਤਾ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
        "offers": "ਮੌਜੂਦਾ ਕੂਪਨ ਜਾਣਕਾਰੀ ਮੇਰੇ ਕੋਲ ਨਹੀਂ ਹੈ। Offers ਭਾਗ ਵੇਖੋ ਜਾਂ ਚੈਕਆਉਟ 'ਤੇ ਕੂਪਨ ਲਗਾਓ।",
        "catalog": "ਸਾਡੇ ਕੈਟਾਲਾਗ ਵਿੱਚ Electronics, Fashion & Apparel, Beauty & Personal Care, Books, Home & Kitchen ਅਤੇ Sports & Fitness ਹਨ।",
        "support": "ਆਰਡਰ ਮਦਦ ਲਈ Account → Help / Contact Support ਖੋਲ੍ਹੋ ਅਤੇ ਆਰਡਰ ID ਦਿਓ। ਤਸਦੀਕ ਕੀਤਾ ਫ਼ੋਨ ਜਾਂ ਈਮੇਲ ਮੇਰੇ ਕੋਲ ਨਹੀਂ ਹੈ।",
        "thanks": "ਜੀ ਆਇਆਂ ਨੂੰ! ਡਿਲਿਵਰੀ, ਵਾਪਸੀ, ਭੁਗਤਾਨ, ਆਰਡਰ ਟ੍ਰੈਕਿੰਗ, ਆਫ਼ਰ, ਕੀਮਤ, ਵਾਰੰਟੀ ਜਾਂ ਗਾਹਕ ਸੇਵਾ ਬਾਰੇ ਪੁੱਛੋ।",
        "fallback": "ਤੁਹਾਡਾ ਸਵਾਲ “{question}” ਸਮਝਿਆ, ਪਰ ਵਿਸ਼ਾ ਪਛਾਣ ਨਹੀਂ ਸਕਿਆ। ਡਿਲਿਵਰੀ/ਥਾਂ, ਵਾਪਸੀ, ਭੁਗਤਾਨ, ਆਰਡਰ ਟ੍ਰੈਕਿੰਗ, ਛੋਟ, ਕੀਮਤ, ਵਾਰੰਟੀ ਜਾਂ ਸਹਾਇਤਾ ਬਾਰੇ ਪੁੱਛੋ।",
        "product_price": "{name} ਦੀ ਕੀਮਤ ₹{price:,} ਹੈ। ਕੀਮਤ ਬਦਲ ਸਕਦੀ ਹੈ; ਨਵੀਂ ਕੀਮਤ ਲਈ ਉਤਪਾਦ ਪੰਨਾ ਵੇਖੋ।",
        "category_products": "{category} ਵਿੱਚ ਉਪਲਬਧ ਚੀਜ਼ਾਂ: {items}।",
    },
    "ur": {
        "location": "ہم بھارت میں ترسیل کرتے ہیں۔ عموماً 3–5 کاروباری دن لگتے ہیں؛ ₹999 سے زیادہ کے آرڈرز پر مفت۔ پتے کی دستیابی یہاں نہیں جانچ سکتا—چیک آؤٹ پر پن کوڈ درج کریں۔",
        "tracking": "میں یہاں لائیو آرڈر نہیں دیکھ سکتا۔ اکاؤنٹ میں سائن ان کرکے Account → My Orders کھولیں۔ آرڈر ID کے ساتھ سپورٹ سے رابطہ کریں۔",
        "offers": "میرے پاس موجودہ کوپن کی معلومات نہیں ہیں۔ Offers سیکشن دیکھیں یا چیک آؤٹ پر کوپن لگائیں۔",
        "catalog": "ہمارے کیٹلاگ میں Electronics، Fashion & Apparel، Beauty & Personal Care، Books، Home & Kitchen اور Sports & Fitness شامل ہیں۔",
        "support": "آرڈر کی مدد کے لیے Account → Help / Contact Support کھولیں اور آرڈر ID دیں۔ میرے پاس تصدیق شدہ فون یا ای میل نہیں ہے۔",
        "thanks": "خوش آمدید! ترسیل، واپسی، ادائیگی، آرڈر ٹریکنگ، آفر، قیمت، وارنٹی یا کسٹمر سپورٹ کے بارے میں پوچھیں۔",
        "fallback": "آپ کا سوال “{question}” سمجھ آیا، مگر موضوع کی شناخت نہیں ہوئی۔ ترسیل/علاقہ، واپسی، ادائیگی، آرڈر ٹریکنگ، رعایت، قیمت، وارنٹی یا سپورٹ کے بارے میں پوچھیں۔",
        "product_price": "{name} کی قیمت ₹{price:,} ہے۔ قیمت بدل سکتی ہے؛ تازہ قیمت کے لیے پروڈکٹ صفحہ دیکھیں۔",
        "category_products": "{category} میں دستیاب اشیا: {items}۔",
    },
    "es": {
        "location": "Realizamos envíos dentro de India. Suelen tardar 3–5 días laborables y son gratis en pedidos superiores a ₹999. No puedo confirmar una dirección aquí; comprueba el código PIN al pagar.",
        "tracking": "No puedo consultar pedidos en tiempo real. Inicia sesión y abre Account → My Orders para ver el estado y el enlace de seguimiento. Contacta con soporte con tu número de pedido.",
        "offers": "No tengo datos actualizados de cupones. Consulta la sección Offers o prueba el cupón al pagar.",
        "catalog": "El catálogo incluye Electronics, Fashion & Apparel, Beauty & Personal Care, Books, Home & Kitchen y Sports & Fitness.",
        "support": "Para ayuda con un pedido, abre Account → Help / Contact Support e incluye el número de pedido. No dispongo de un teléfono o correo verificado.",
        "thanks": "¡De nada! Pregúntame por envíos, devoluciones, pagos, seguimiento, ofertas, precios, garantías o atención al cliente.",
        "fallback": "Entendí tu pregunta «{question}», pero no pude identificar el tema. Pregunta por envíos o zonas, devoluciones, pagos, seguimiento, descuentos, precios, garantías o soporte.",
        "product_price": "{name} cuesta ₹{price:,}. El precio puede cambiar; consulta la página del producto para ver el precio actual.",
        "category_products": "Productos disponibles en {category}: {items}.",
    },
    "fr": {
        "location": "Nous livrons en Inde. Le délai habituel est de 3 à 5 jours ouvrés, et la livraison est gratuite au-delà de ₹999. Je ne peux pas vérifier une adresse ici ; saisissez le code PIN au paiement.",
        "tracking": "Je ne peux pas consulter les commandes en direct. Connectez-vous et ouvrez Account → My Orders pour voir le statut et le suivi. Contactez l’assistance avec votre numéro de commande.",
        "offers": "Je n’ai pas accès aux coupons en cours. Consultez la rubrique Offers ou essayez votre coupon au paiement.",
        "catalog": "Notre catalogue comprend Electronics, Fashion & Apparel, Beauty & Personal Care, Books, Home & Kitchen et Sports & Fitness.",
        "support": "Pour une commande, ouvrez Account → Help / Contact Support et indiquez son numéro. Je ne dispose pas de téléphone ni d’adresse e-mail vérifiés.",
        "thanks": "Avec plaisir ! Demandez-moi des informations sur la livraison, les retours, les paiements, le suivi, les offres, les prix, la garantie ou l’assistance.",
        "fallback": "J’ai compris votre question « {question} », mais je n’ai pas identifié le sujet. Vous pouvez demander la livraison, les retours, les paiements, le suivi, les réductions, les prix, la garantie ou l’assistance.",
        "product_price": "{name} coûte ₹{price:,}. Le prix peut changer ; consultez la page du produit pour le prix actuel.",
        "category_products": "Articles disponibles dans {category} : {items}.",
    },
    "de": {
        "location": "Wir liefern innerhalb Indiens. Die Lieferung dauert meist 3–5 Werktage und ist ab ₹999 kostenlos. Eine konkrete Adresse kann ich hier nicht prüfen; geben Sie zur Prüfung die PIN beim Checkout ein.",
        "tracking": "Ich kann keine Live-Bestellungen abrufen. Melden Sie sich an und öffnen Sie Account → My Orders für Status und Sendungsverfolgung. Wenden Sie sich mit der Bestellnummer an den Support.",
        "offers": "Aktuelle Gutscheine kann ich nicht abrufen. Prüfen Sie den Bereich Offers oder lösen Sie den Gutschein beim Checkout ein.",
        "catalog": "Unser Sortiment umfasst Electronics, Fashion & Apparel, Beauty & Personal Care, Books, Home & Kitchen und Sports & Fitness.",
        "support": "Bei Fragen zur Bestellung öffnen Sie Account → Help / Contact Support und geben Sie die Bestellnummer an. Eine bestätigte Telefonnummer oder E-Mail liegt mir nicht vor.",
        "thanks": "Gern! Fragen Sie mich zu Lieferung, Rückgabe, Zahlung, Sendungsverfolgung, Angeboten, Preisen, Garantie oder Kundenservice.",
        "fallback": "Ich habe Ihre Frage „{question}“ verstanden, konnte sie aber keinem Thema zuordnen. Fragen Sie nach Lieferung, Rückgabe, Zahlung, Sendungsverfolgung, Rabatten, Preisen, Garantie oder Support.",
        "product_price": "{name} kostet ₹{price:,}. Preise können sich ändern; den aktuellen Preis finden Sie auf der Produktseite.",
        "category_products": "Produkte in {category}: {items}.",
    },
    "ja": {
        "location": "インド国内へ配送しています。通常3〜5営業日で、₹999を超える注文は送料無料です。住所の配送可否はここでは確認できません。決済時にPINコードを入力してください。",
        "tracking": "ここでは注文のリアルタイム情報を確認できません。ログインして Account → My Orders を開くと状況と追跡リンクを確認できます。",
        "offers": "現在のクーポン情報は確認できません。Offers セクションをご覧いただくか、決済時にクーポンをお試しください。",
        "catalog": "商品カテゴリは Electronics、Fashion & Apparel、Beauty & Personal Care、Books、Home & Kitchen、Sports & Fitness です。",
        "support": "注文については Account → Help / Contact Support を開き、注文IDを添えてお問い合わせください。確認済みの電話番号やメールアドレスはありません。",
        "thanks": "どういたしまして。配送、返品、支払い、注文追跡、セール、価格、保証、サポートについてお尋ねください。",
        "fallback": "「{question}」というご質問ですね。内容を特定できませんでした。配送、返品、支払い、追跡、割引、価格、保証、サポートについてお尋ねください。",
        "product_price": "{name}の価格は₹{price:,}です。価格は変動するため、最新価格は商品ページをご確認ください。",
        "category_products": "{category}の商品: {items}。",
    },
    "ar": {
        "location": "نوصل داخل الهند. يستغرق التوصيل عادةً من 3 إلى 5 أيام عمل، وهو مجاني للطلبات التي تزيد على ₹999. لا يمكنني التحقق من عنوان محدد هنا؛ أدخل الرمز البريدي عند الدفع.",
        "tracking": "لا أستطيع الوصول إلى الطلبات مباشرةً. سجّل الدخول وافتح Account → My Orders لمعرفة الحالة ورابط التتبع. تواصل مع الدعم برقم الطلب.",
        "offers": "لا تتوفر لدي معلومات مباشرة عن القسائم الحالية. راجع قسم Offers أو جرّب القسيمة عند الدفع.",
        "catalog": "تشمل فئاتنا Electronics وFashion & Apparel وBeauty & Personal Care وBooks وHome & Kitchen وSports & Fitness.",
        "support": "للمساعدة بشأن طلب، افتح Account → Help / Contact Support وأرفق رقم الطلب. ليس لدي رقم هاتف أو بريد إلكتروني موثّق.",
        "thanks": "على الرحب والسعة! اسأل عن التوصيل أو الإرجاع أو الدفع أو تتبع الطلب أو العروض أو الأسعار أو الضمان أو الدعم.",
        "fallback": "فهمت سؤالك «{question}»، لكن لم أتمكن من تحديد موضوعه. اسأل عن التوصيل أو المناطق أو الإرجاع أو الدفع أو التتبع أو الخصومات أو الأسعار أو الضمان أو الدعم.",
        "product_price": "سعر {name} هو ₹{price:,}. قد يتغير السعر؛ راجع صفحة المنتج للسعر الأحدث.",
        "category_products": "المنتجات المتاحة ضمن {category}: {items}.",
    },
})

CATEGORY_KEYWORDS = {
    "Electronics": ("electronics", "electronic", "எலக்ட்ரானிக்ஸ்", "ఎలక్ట్రానిక్స్", "ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್", "इलेक्ट्रॉनिक्स", "ইলেকট্রনিক্স", "ઇલેક્ટ્રોનિક્સ", "الکترونکس", "إلكترونيات", "elektronik", "électronique", "electrónica", "電子機器", "ഇലക്ട്രോണിക്സ്"),
    "Fashion & Apparel": ("fashion", "apparel", "clothing", "clothes", "ಫ್ಯಾಷನ್", "ಬಟ್ಟೆ", "ಉಡುಪು", "फैशन", "कपड़े", "ஆடை", "உடை", "దుస్తులు", "পোশাক", "કપડાં", "ਕੱਪੜੇ", "لباس", "ropa", "vêtements", "kleidung", "洋服", "വസ്ത്രം"),
    "Home & Kitchen": ("home", "kitchen", "cookware", "ಮನೆ", "ಅಡುಗೆಮನೆ", "கிச்சன்", "வீடு", "வீட்டு", "ఇల్లు", "వంటగది", "घर", "रसोई", "বাড়ি", "ઘર", "ਘਰ", "گھر", "cocina", "maison", "küche", "キッチン", "അടുക്കള"),
    "Beauty & Personal Care": ("beauty", "personal care", "skincare", "skin care", "hair care", "ಸೌಂದರ್ಯ", "ವೈಯಕ್ತಿಕ ಆರೈಕೆ", "அழகு", "அழகுப் பராமரிப்பு", "అందం", "వ్యక్తిగత సంరక్షణ", "सौंदर्य", "त्वचा", "সৌন্দর্য", "સૌંદર્ય", "خوبصورتی", "belleza", "beauté", "pflege", "美容", "സൗന്ദര്യം"),
    "Sports & Fitness": ("sports", "fitness", "workout", "gym", "yoga", "ಕ್ರೀಡೆ", "ಫಿಟ್‌ನೆಸ್", "உடற்பயிற்சி", "விளையாட்டு", "క్రీడలు", "ఫిట్‌నెస్", "खेल", "व्यायाम", "খেলাধুলা", "રમતગમત", "کھیل", "deporte", "sport", "رياضة", "運動", "കായികം"),
    "Books": ("books", "book", "ಪುಸ್ತಕ", "ಪುಸ್ತಕಗಳು", "किताब", "पुस्तक", "புத்தகம்", "புத்தகங்கள்", "పుస్తకం", "పుస్తకాలు", "বই", "પુસ્તકો", "کتابیں", "libros", "livres", "bücher", "本", "പുസ്തകം"),
}

PRODUCT_KEYWORDS = {
    "Nova X Pro Smartphone": ("smartphone", "mobile phone", "5g phone"),
    "PixelView 4K Smart TV": ("smart tv", "4k tv", "television"),
    "SonicBeat Wireless Headphones": ("headphones", "wireless headphones", "noise cancelling headphones"),
    "ThunderMax Bluetooth Speaker": ("bluetooth speaker", "portable speaker"),
    "Everyday Cotton Kurta": ("cotton kurta", "kurta"),
    "Classic Denim Jacket": ("denim jacket", "jean jacket"),
    "FlexRun Running Shoes": ("running shoes", "trainers"),
    "Everyday Canvas Backpack": ("canvas backpack", "backpack", "rucksack"),
    "BrewMate French Press": ("french press", "coffee maker"),
    "ChefPro Nonstick Cookware Set": ("cookware", "nonstick cookware", "cookware set"),
    "CozyNest Cotton Bedsheet": ("bedsheet", "bed sheet", "bedsheet set"),
    "GlowPure Vitamin C Face Serum": ("face serum", "vitamin c serum", "serum"),
    "SilkShine Hair Care Kit": ("hair care kit", "shampoo", "conditioner"),
    "FreshBloom Eau de Parfum": ("perfume", "eau de parfum", "fragrance"),
    "PulseFit Smartwatch": ("smartwatch", "fitness watch"),
    "PowerFlex Yoga Mat": ("yoga mat", "exercise mat"),
    "ProGrip Adjustable Dumbbells": ("dumbbells", "adjustable dumbbell", "weights"),
    "The Quiet Horizon": ("the quiet horizon", "fiction novel"),
    "Everyday Personal Finance": ("personal finance book", "finance book"),
    "The Curious Science Handbook": ("science handbook", "science book"),
}

WARRANTY_RESPONSES = {
    "en": "Most electronics have a 1-year manufacturer warranty. Repair or replacement depends on the product's warranty terms and inspection; check its listing and contact support with your order ID.",
    "kn": "ಹೆಚ್ಚಿನ ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್‌ಗೆ ತಯಾರಕರ 1 ವರ್ಷದ ಖಾತರಿ ಇರುತ್ತದೆ. ದುರಸ್ತಿ ಅಥವಾ ಬದಲಿ ಉತ್ಪನ್ನದ ಖಾತರಿ ನಿಯಮಗಳು ಮತ್ತು ಪರಿಶೀಲನೆಗೆ ಒಳಪಟ್ಟಿರುತ್ತದೆ; ಆರ್ಡರ್ ಐಡಿಯೊಂದಿಗೆ ಬೆಂಬಲವನ್ನು ಸಂಪರ್ಕಿಸಿ.",
    "hi": "अधिकतर इलेक्ट्रॉनिक्स पर निर्माता की 1 साल की वारंटी होती है। मरम्मत या बदलना उत्पाद की शर्तों और जाँच पर निर्भर है; ऑर्डर ID के साथ सहायता से संपर्क करें।",
    "ta": "பெரும்பாலான மின்னணுப் பொருட்களுக்கு உற்பத்தியாளரின் 1 ஆண்டு உத்தரவாதம் உண்டு. பழுது அல்லது மாற்றம் தயாரிப்பு விதிமுறைகள் மற்றும் பரிசோதனைக்கு உட்பட்டது; ஆர்டர் ID-யுடன் ஆதரவைத் தொடர்புகொள்ளுங்கள்.",
    "te": "చాలా ఎలక్ట్రానిక్స్‌కు తయారీదారు 1 సంవత్సరం వారంటీ ఉంటుంది. మరమ్మత్తు లేదా మార్పిడి ఉత్పత్తి నిబంధనలు, తనిఖీపై ఆధారపడి ఉంటుంది; ఆర్డర్ IDతో సపోర్ట్‌ను సంప్రదించండి.",
    "ml": "മിക്ക ഇലക്ട്രോണിക്സ് ഉൽപ്പന്നങ്ങൾക്കും നിർമ്മാതാവിന്റെ 1 വർഷത്തെ വാറന്റിയുണ്ട്. അറ്റകുറ്റപ്പണിയോ മാറ്റമോ ഉൽപ്പന്ന നിബന്ധനകളും പരിശോധനയും അനുസരിച്ചിരിക്കും; ഓർഡർ ഐഡിയുമായി പിന്തുണയെ ബന്ധപ്പെടുക.",
    "mr": "बहुतेक इलेक्ट्रॉनिक्सवर उत्पादकाची 1 वर्षाची वॉरंटी असते. दुरुस्ती किंवा बदल उत्पादनाच्या अटी आणि तपासणीवर अवलंबून आहे; ऑर्डर ID सह सहाय्याशी संपर्क साधा.",
    "bn": "বেশিরভাগ ইলেকট্রনিক পণ্যে প্রস্তুতকারকের 1 বছরের ওয়ারেন্টি থাকে। মেরামত বা বদল পণ্যের শর্ত ও পরীক্ষার ওপর নির্ভর করে; অর্ডার ID দিয়ে সহায়তায় যোগাযোগ করুন।",
    "gu": "મોટાભાગના ઇલેક્ટ્રોનિક્સ પર ઉત્પાદકની 1 વર્ષની વોરંટી હોય છે. સમારકામ અથવા બદલવું ઉત્પાદનની શરતો અને તપાસ પર આધારિત છે; ઓર્ડર ID સાથે સહાયનો સંપર્ક કરો.",
    "pa": "ਜ਼ਿਆਦਾਤਰ ਇਲੈਕਟ੍ਰਾਨਿਕ ਚੀਜ਼ਾਂ ਉੱਤੇ ਨਿਰਮਾਤਾ ਦੀ 1 ਸਾਲ ਦੀ ਵਾਰੰਟੀ ਹੁੰਦੀ ਹੈ। ਮੁਰੰਮਤ ਜਾਂ ਬਦਲੀ ਉਤਪਾਦ ਦੀਆਂ ਸ਼ਰਤਾਂ ਅਤੇ ਜਾਂਚ ਉੱਤੇ ਨਿਰਭਰ ਕਰਦੀ ਹੈ; ਆਰਡਰ ID ਨਾਲ ਸਹਾਇਤਾ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
    "ur": "زیادہ تر الیکٹرانکس پر کمپنی کی 1 سال کی وارنٹی ہوتی ہے۔ مرمت یا تبدیلی مصنوعات کی شرائط اور جانچ پر منحصر ہے؛ آرڈر ID کے ساتھ سپورٹ سے رابطہ کریں۔",
    "es": "La mayoría de los productos electrónicos incluye 1 año de garantía del fabricante. La reparación o sustitución depende de las condiciones y la revisión; contacta con soporte con tu número de pedido.",
    "fr": "La plupart des appareils électroniques bénéficient d’un an de garantie fabricant. La réparation ou le remplacement dépend des conditions et d’une vérification ; contactez l’assistance avec votre numéro de commande.",
    "de": "Die meisten Elektronikartikel haben eine einjährige Herstellergarantie. Reparatur oder Ersatz hängen von den Garantiebedingungen und einer Prüfung ab; wenden Sie sich mit der Bestellnummer an den Support.",
    "ja": "ほとんどの電子機器にはメーカーの1年間保証が付いています。修理や交換は保証条件と確認結果によります。注文IDを添えてサポートへお問い合わせください。",
    "ar": "تأتي معظم الأجهزة الإلكترونية مع ضمان من الشركة المصنعة لمدة سنة. يعتمد الإصلاح أو الاستبدال على شروط المنتج والفحص؛ تواصل مع الدعم برقم الطلب.",
}

PAYMENT_RESPONSES = {
    "en": "We accept UPI payments through apps such as Google Pay (GPay), PhonePe, and Paytm, plus credit/debit cards, net banking, and cash on delivery where available.",
    "kn": "Google Pay (GPay), PhonePe ಮತ್ತು Paytm ಮೂಲಕ UPI, ಕ್ರೆಡಿಟ್/ಡೆಬಿಟ್ ಕಾರ್ಡ್, ನೆಟ್ ಬ್ಯಾಂಕಿಂಗ್ ಹಾಗೂ ಲಭ್ಯವಿದ್ದಲ್ಲಿ ಕ್ಯಾಶ್ ಆನ್ ಡೆಲಿವರಿಯನ್ನು ಸ್ವೀಕರಿಸುತ್ತೇವೆ.",
    "hi": "हम Google Pay (GPay), PhonePe और Paytm जैसे ऐप से UPI, क्रेडिट/डेबिट कार्ड, नेट बैंकिंग और जहाँ उपलब्ध हो कैश ऑन डिलीवरी स्वीकार करते हैं।",
    "ta": "Google Pay (GPay), PhonePe, Paytm வழியாக UPI, கிரெடிட்/டெபிட் கார்டு, நெட் பேங்கிங் மற்றும் கிடைக்கும் இடங்களில் கேஷ் ஆன் டெலிவரியை ஏற்கிறோம்.",
    "te": "Google Pay (GPay), PhonePe, Paytm ద్వారా UPI, క్రెడిట్/డెబిట్ కార్డులు, నెట్ బ్యాంకింగ్ మరియు అందుబాటులో ఉన్న చోట క్యాష్ ఆన్ డెలివరీని అంగీకరిస్తాం.",
    "ml": "Google Pay (GPay), PhonePe, Paytm വഴിയുള്ള UPI, ക്രെഡിറ്റ്/ഡെബിറ്റ് കാർഡുകൾ, നെറ്റ് ബാങ്കിംഗ്, ലഭ്യമായിടത്ത് ക്യാഷ് ഓൺ ഡെലിവറി എന്നിവ സ്വീകരിക്കുന്നു.",
    "mr": "Google Pay (GPay), PhonePe आणि Paytm द्वारे UPI, क्रेडिट/डेबिट कार्ड, नेट बँकिंग आणि उपलब्ध ठिकाणी कॅश ऑन डिलिव्हरी स्वीकारतो.",
    "bn": "Google Pay (GPay), PhonePe ও Paytm-এর মাধ্যমে UPI, ক্রেডিট/ডেবিট কার্ড, নেট ব্যাংকিং এবং যেখানে উপলভ্য ক্যাশ অন ডেলিভারি গ্রহণ করি।",
    "gu": "Google Pay (GPay), PhonePe અને Paytm દ્વારા UPI, ક્રેડિટ/ડેબિટ કાર્ડ, નેટ બેંકિંગ અને જ્યાં ઉપલબ્ધ હોય ત્યાં કેશ ઑન ડિલિવરી સ્વીકારીએ છીએ.",
    "pa": "Google Pay (GPay), PhonePe ਅਤੇ Paytm ਰਾਹੀਂ UPI, ਕ੍ਰੈਡਿਟ/ਡੈਬਿਟ ਕਾਰਡ, ਨੈੱਟ ਬੈਂਕਿੰਗ ਅਤੇ ਜਿੱਥੇ ਉਪਲਬਧ ਹੋਵੇ ਕੈਸ਼ ਆਨ ਡਿਲਿਵਰੀ ਸਵੀਕਾਰ ਕਰਦੇ ਹਾਂ।",
    "ur": "ہم Google Pay (GPay)، PhonePe اور Paytm کے ذریعے UPI، کریڈٹ/ڈیبٹ کارڈ، نیٹ بینکنگ اور جہاں دستیاب ہو کیش آن ڈیلیوری قبول کرتے ہیں۔",
    "es": "Aceptamos UPI mediante Google Pay (GPay), PhonePe y Paytm, además de tarjetas de crédito/débito, banca electrónica y pago contra reembolso donde esté disponible.",
    "fr": "Nous acceptons UPI via Google Pay (GPay), PhonePe et Paytm, ainsi que les cartes bancaires, les virements et le paiement à la livraison lorsqu’il est disponible.",
    "de": "Wir akzeptieren UPI über Google Pay (GPay), PhonePe und Paytm sowie Kredit-/Debitkarten, Online-Banking und, sofern verfügbar, Nachnahme.",
    "ja": "Google Pay (GPay)、PhonePe、PaytmによるUPI、クレジット/デビットカード、ネットバンキング、および利用可能な地域では代金引換をご利用いただけます。",
    "ar": "نقبل UPI عبر Google Pay (GPay) وPhonePe وPaytm، إضافة إلى بطاقات الائتمان والخصم والخدمات المصرفية عبر الإنترنت والدفع عند الاستلام حيثما كان متاحاً.",
}

LOCATION_FOLLOWUPS = {
    "en": "For {location}, enter your PIN code at checkout to confirm delivery availability.",
    "kn": "{location}ಗೆ ಡೆಲಿವರಿ ಲಭ್ಯತೆಯನ್ನು ಖಚಿತಪಡಿಸಲು ಚೆಕ್‌ಔಟ್‌ನಲ್ಲಿ ಪಿನ್ ಕೋಡ್ ನಮೂದಿಸಿ.",
    "hi": "{location} में डिलीवरी की पुष्टि के लिए चेकआउट पर पिन कोड डालें।",
    "ta": "{location} விநியோகத்தை உறுதிப்படுத்த செக் அவுட்டில் பின் குறியீட்டை உள்ளிடவும்.",
    "te": "{location} డెలివరీని నిర్ధారించడానికి చెక్అవుట్‌లో పిన్ కోడ్ నమోదు చేయండి.",
    "ml": "{location}-ലേക്കുള്ള ഡെലിവറി ഉറപ്പാക്കാൻ ചെക്ക്ഔട്ടിൽ പിൻകോഡ് നൽകുക.",
    "mr": "{location} डिलिव्हरीची खात्री करण्यासाठी चेकआउटमध्ये पिन कोड द्या.",
    "bn": "{location}-এ ডেলিভারি নিশ্চিত করতে চেকআউটে পিন কোড দিন।",
    "gu": "{location} ડિલિવરીની ખાતરી કરવા ચેકઆઉટમાં પિન કોડ દાખલ કરો.",
    "pa": "{location} ਡਿਲਿਵਰੀ ਦੀ ਪੁਸ਼ਟੀ ਲਈ ਚੈਕਆਉਟ 'ਤੇ ਪਿਨ ਕੋਡ ਦਿਓ।",
    "ur": "{location} میں ترسیل کی تصدیق کے لیے چیک آؤٹ پر پن کوڈ درج کریں۔",
    "es": "Para {location}, introduce tu código PIN al pagar y confirma la disponibilidad.",
    "fr": "Pour {location}, saisissez votre code PIN au paiement pour confirmer la disponibilité.",
    "de": "Geben Sie für {location} beim Checkout Ihre PIN ein, um die Lieferbarkeit zu prüfen.",
    "ja": "{location}への配送可否は、決済時にPINコードを入力してご確認ください。",
    "ar": "لتأكيد التوصيل إلى {location}، أدخل الرمز البريدي عند الدفع.",
}

PRICE_PROMPTS = {
    "en": "Which product or category are you asking about? Name it and I can show its catalog price in ₹.",
    "kn": "ಯಾವ ಉತ್ಪನ್ನ ಅಥವಾ ವಿಭಾಗದ ಬೆಲೆ ಬೇಕು? ಹೆಸರನ್ನು ತಿಳಿಸಿ; ಅದರ ಬೆಲೆಯನ್ನು ₹ಗಳಲ್ಲಿ ಹೇಳುತ್ತೇನೆ.",
    "hi": "आप किस उत्पाद या श्रेणी की कीमत जानना चाहते हैं? नाम बताइए, मैं ₹ में कीमत बताऊँगा।",
    "ta": "எந்தப் பொருள் அல்லது பிரிவின் விலை வேண்டும்? பெயரைக் கூறுங்கள்; ₹ விலையைச் சொல்கிறேன்.",
    "te": "ఏ ఉత్పత్తి లేదా విభాగం ధర కావాలి? పేరు చెప్పండి; ₹ ధరను తెలియజేస్తాను.",
    "ml": "ഏത് ഉൽപ്പന്നത്തിന്റെയോ വിഭാഗത്തിന്റെയോ വിലയാണ് വേണ്ടത്? പേര് പറയൂ; ₹ വില അറിയിക്കാം.",
    "mr": "कोणत्या उत्पादनाची किंवा विभागाची किंमत हवी आहे? नाव सांगा; ₹ मधील किंमत सांगतो.",
    "bn": "কোন পণ্য বা বিভাগের দাম জানতে চান? নাম বলুন, ₹-তে দাম জানাব।",
    "gu": "કયા ઉત્પાદન અથવા વિભાગની કિંમત જાણવી છે? નામ જણાવો, હું ₹માં કિંમત કહીશ.",
    "pa": "ਕਿਸ ਉਤਪਾਦ ਜਾਂ ਸ਼੍ਰੇਣੀ ਦੀ ਕੀਮਤ ਜਾਣਨੀ ਹੈ? ਨਾਮ ਦੱਸੋ, ਮੈਂ ₹ ਵਿੱਚ ਕੀਮਤ ਦੱਸਾਂਗਾ।",
    "ur": "آپ کس چیز یا زمرے کی قیمت جاننا چاہتے ہیں؟ نام بتائیں، میں ₹ میں قیمت بتاؤں گا۔",
    "es": "¿Qué producto o categoría buscas? Dime el nombre y te indicaré el precio en ₹.",
    "fr": "Quel produit ou quelle catégorie recherchez-vous ? Indiquez son nom pour obtenir le prix en ₹.",
    "de": "Welchen Artikel oder welche Kategorie meinen Sie? Nennen Sie den Namen, dann zeige ich den Preis in ₹.",
    "ja": "どの商品またはカテゴリの価格をお探しですか？名前を教えていただければ₹でご案内します。",
    "ar": "ما المنتج أو الفئة التي تريد معرفة سعرها؟ اذكر الاسم وسأعرض السعر بالروبية ₹.",
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


CHATBOT_ANSWER_BANK = {
    "services": {
        "en": "We provide AI product recommendations, order tracking, fast India-wide delivery, 7-day returns, secure payments, warranty help, and fake-review checks.",
        "kn": "AI ಉತ್ಪನ್ನ ಶಿಫಾರಸು, ಆರ್ಡರ್ ಟ್ರ್ಯಾಕಿಂಗ್, ಭಾರತದಾದ್ಯಂತ ವೇಗದ ಡೆಲಿವರಿ, 7 ದಿನಗಳ ರಿಟರ್ನ್, ಸುರಕ್ಷಿತ ಪಾವತಿ, ವಾರಂಟಿ ಸಹಾಯ ಮತ್ತು ನಕಲಿ ವಿಮರ್ಶೆ ಪರಿಶೀಲನೆ ಒದಗಿಸುತ್ತೇವೆ.",
        "hi": "हम AI उत्पाद सुझाव, ऑर्डर ट्रैकिंग, पूरे भारत में तेज़ डिलीवरी, 7-दिन रिटर्न, सुरक्षित भुगतान, वारंटी सहायता और नकली समीक्षा जाँच देते हैं।",
        "ta": "AI தயாரிப்பு பரிந்துரைகள், ஆர்டர் கண்காணிப்பு, இந்தியா முழுவதும் விரைவு விநியோகம், 7 நாள் திரும்பப்பெறுதல், பாதுகாப்பான கட்டணம், உத்தரவாத உதவி, போலி மதிப்புரை சரிபார்ப்பு வழங்குகிறோம்.",
        "te": "AI ఉత్పత్తి సిఫార్సులు, ఆర్డర్ ట్రాకింగ్, భారత్ అంతటా వేగవంతమైన డెలివరీ, 7 రోజుల రిటర్న్, సురక్షిత చెల్లింపులు, వారంటీ సహాయం, నకిలీ సమీక్ష తనిఖీ అందిస్తాం.",
    },
    "delivery": {
        "en": "We deliver across India, including Mysore, Bengaluru, and Karnataka. Delivery usually takes 2–4 days.",
        "kn": "ಮೈಸೂರು, ಬೆಂಗಳೂರು ಮತ್ತು ಕರ್ನಾಟಕ ಸೇರಿದಂತೆ ಭಾರತದಾದ್ಯಂತ ಡೆಲಿವರಿ ಮಾಡುತ್ತೇವೆ. ಸಾಮಾನ್ಯವಾಗಿ 2–4 ದಿನಗಳಲ್ಲಿ ತಲುಪುತ್ತದೆ.",
        "hi": "हम मैसूर, बेंगलुरु और कर्नाटक सहित पूरे भारत में डिलीवरी करते हैं। आमतौर पर 2–4 दिन लगते हैं।",
        "ta": "மைசூர், பெங்களூரு, கர்நாடகா உட்பட இந்தியா முழுவதும் விநியோகம் செய்கிறோம். வழக்கமாக 2–4 நாட்களில் கிடைக்கும்.",
        "te": "మైసూరు, బెంగళూరు, కర్ణాటకతో సహా భారత్ అంతటా డెలివరీ చేస్తాం. సాధారణంగా 2–4 రోజుల్లో చేరుతుంది.",
    },
    "shipping_fee": {
        "en": "Shipping is free on orders above ₹499. Orders of ₹499 or less may have a delivery fee shown at checkout.",
        "kn": "₹499 క್ಕಿಂತ ಹೆಚ್ಚಿನ ಆರ್ಡರ್‌ಗಳಿಗೆ ಡೆಲಿವರಿ ಉಚಿತ. ₹499 ಅಥವಾ ಕಡಿಮೆ ಆರ್ಡರ್‌ಗಳ ಶುಲ್ಕ ಚೆಕ್‌ಔಟ್‌ನಲ್ಲಿ ಕಾಣುತ್ತದೆ.",
        "hi": "₹499 से अधिक के ऑर्डर पर डिलीवरी मुफ़्त है। ₹499 या उससे कम के ऑर्डर का शुल्क चेकआउट पर दिखेगा।",
        "ta": "₹499-க்கு மேற்பட்ட ஆர்டர்களுக்கு விநியோகம் இலவசம். ₹499 அல்லது அதற்குக் குறைவான ஆர்டர்களுக்கான கட்டணம் செக் அவுட்டில் காட்டப்படும்.",
        "te": "₹499 కంటే ఎక్కువ ఆర్డర్లకు డెలివరీ ఉచితం. ₹499 లేదా అంతకంటే తక్కువ ఆర్డర్లకు రుసుము చెక్అవుట్‌లో చూపబడుతుంది.",
    },
    "returns": {
        "en": "Eligible products can be returned within 7 days of delivery, with no questions asked. Schedule a pickup from My Orders.",
        "kn": "ಅರ್ಹ ಉತ್ಪನ್ನಗಳನ್ನು ಡೆಲಿವರಿ ಆದ 7 ದಿನಗಳೊಳಗೆ ಯಾವುದೇ ಪ್ರಶ್ನೆಯಿಲ್ಲದೆ ಹಿಂತಿರುಗಿಸಬಹುದು. My Orders ನಲ್ಲಿ ಪಿಕಪ್ ನಿಗದಿಪಡಿಸಿ.",
        "hi": "पात्र उत्पाद डिलीवरी के 7 दिनों के भीतर बिना सवाल लौटाए जा सकते हैं। My Orders में पिकअप तय करें।",
        "ta": "தகுதியுள்ள பொருட்களை விநியோகித்த 7 நாட்களுக்குள் கேள்வியின்றி திருப்பலாம். My Orders-இல் பிக்கப்பை திட்டமிடுங்கள்.",
        "te": "అర్హమైన ఉత్పత్తులను డెలివరీ అయిన 7 రోజుల్లో ఎలాంటి ప్రశ్నలు లేకుండా తిరిగి ఇవ్వవచ్చు. My Ordersలో పికప్‌ను ఏర్పాటు చేయండి.",
    },
    "exchange": {
        "en": "Eligible items can be exchanged within 7 days of delivery. Start an exchange request in My Orders.",
        "kn": "ಅರ್ಹ ವಸ್ತುಗಳನ್ನು ಡೆಲಿವರಿ ಆದ 7 ದಿನಗಳಲ್ಲಿ ಬದಲಾಯಿಸಬಹುದು. My Orders ನಲ್ಲಿ ವಿನಿಮಯ ವಿನಂತಿ ಸಲ್ಲಿಸಿ.",
        "hi": "पात्र वस्तुएँ डिलीवरी के 7 दिनों के भीतर बदली जा सकती हैं। My Orders में अनुरोध शुरू करें।",
        "ta": "தகுதியுள்ள பொருட்களை விநியோகித்த 7 நாட்களுக்குள் மாற்றலாம். My Orders-இல் கோரிக்கையைத் தொடங்குங்கள்.",
        "te": "అర్హమైన వస్తువులను డెలివరీ అయిన 7 రోజుల్లో మార్చుకోవచ్చు. My Ordersలో అభ్యర్థన ప్రారంభించండి.",
    },
    "refund": {
        "en": "After a returned item is picked up and checked, the refund is issued to the original payment method within 3–5 business days.",
        "kn": "ವಸ್ತುವನ್ನು ಪಿಕಪ್ ಮಾಡಿ ಪರಿಶೀಲಿಸಿದ ನಂತರ, ರಿಫಂಡ್ ಮೂಲ ಪಾವತಿ ವಿಧಾನಕ್ಕೆ 3–5 ಕೆಲಸದ ದಿನಗಳಲ್ಲಿ ಜಮೆಯಾಗುತ್ತದೆ.",
        "hi": "वापसी की वस्तु पिकअप और जाँच के बाद रिफंड मूल भुगतान माध्यम में 3–5 कार्यदिवस में जमा होता है।",
        "ta": "திருப்பிய பொருள் பிக்கப் செய்து சரிபார்த்த பின், அசல் கட்டண முறைக்கு 3–5 வேலை நாட்களில் பணம் திரும்பும்.",
        "te": "రిటర్న్ వస్తువును పికప్ చేసి తనిఖీ చేసిన తర్వాత, అసలు చెల్లింపు పద్ధతికి 3–5 పని దినాల్లో రీఫండ్ జమ అవుతుంది.",
    },
    "payments": {
        "en": "We accept UPI, Google Pay (GPay), PhonePe, Paytm, credit/debit cards, net banking, and Cash on Delivery where available.",
        "kn": "UPI, Google Pay (GPay), PhonePe, Paytm, ಕ್ರೆಡಿಟ್/ಡೆಬಿಟ್ ಕಾರ್ಡ್, ನೆಟ್ ಬ್ಯಾಂಕಿಂಗ್ ಮತ್ತು ಲಭ್ಯವಿದ್ದಲ್ಲಿ Cash on Delivery ಸ್ವೀಕರಿಸುತ್ತೇವೆ.",
        "hi": "हम UPI, Google Pay (GPay), PhonePe, Paytm, क्रेडिट/डेबिट कार्ड, नेट बैंकिंग और उपलब्धता के अनुसार Cash on Delivery स्वीकार करते हैं।",
        "ta": "UPI, Google Pay (GPay), PhonePe, Paytm, கிரெடிட்/டெபிட் கார்டு, நெட் பேங்கிங் மற்றும் கிடைக்கும் இடங்களில் Cash on Delivery ஏற்கிறோம்.",
        "te": "UPI, Google Pay (GPay), PhonePe, Paytm, క్రెడిట్/డెబిట్ కార్డులు, నెట్ బ్యాంకింగ్, అందుబాటులో ఉన్న చోట Cash on Delivery అంగీకరిస్తాం.",
    },
    "tracking": {
        "en": "Sign in, open My Orders, and enter or select your Order ID to view its latest status and tracking link.",
        "kn": "ಸೈನ್ ಇನ್ ಮಾಡಿ, My Orders ತೆರೆಯಿರಿ ಮತ್ತು Order ID ಆಯ್ಕೆ ಮಾಡಿ; ಇತ್ತೀಚಿನ ಸ್ಥಿತಿ ಹಾಗೂ ಟ್ರ್ಯಾಕಿಂಗ್ ಲಿಂಕ್ ಕಾಣುತ್ತದೆ.",
        "hi": "साइन इन करें, My Orders खोलें और Order ID चुनें। वहाँ नवीनतम स्थिति और ट्रैकिंग लिंक दिखेगा।",
        "ta": "உள்நுழைந்து My Orders திறந்து Order ID-ஐத் தேர்ந்தெடுக்கவும். சமீபத்திய நிலையும் கண்காணிப்பு இணைப்பும் அங்கே இருக்கும்.",
        "te": "సైన్ ఇన్ చేసి My Orders తెరిచి Order IDని ఎంచుకోండి. తాజా స్థితి, ట్రాకింగ్ లింక్ అక్కడ కనిపిస్తాయి.",
    },
    "cancel": {
        "en": "You can cancel an order free of charge any time before it is dispatched. Open My Orders and select Cancel.",
        "kn": "ಆರ್ಡರ್ ರವಾನೆಯಾಗುವ ಮೊದಲು ಯಾವುದೇ ಸಮಯದಲ್ಲಿ ಉಚಿತವಾಗಿ ರದ್ದುಮಾಡಬಹುದು. My Orders ತೆರೆಯಿರಿ ಮತ್ತು Cancel ಆಯ್ಕೆಮಾಡಿ.",
        "hi": "ऑर्डर भेजे जाने से पहले कभी भी मुफ़्त रद्द कर सकते हैं। My Orders खोलकर Cancel चुनें।",
        "ta": "ஆர்டர் அனுப்பப்படுவதற்கு முன் எப்போது வேண்டுமானாலும் இலவசமாக ரத்து செய்யலாம். My Orders-இல் Cancel தேர்ந்தெடுக்கவும்.",
        "te": "ఆర్డర్ పంపే ముందు ఎప్పుడైనా ఉచితంగా రద్దు చేయవచ్చు. My Orders తెరిచి Cancel ఎంచుకోండి.",
    },
    "offers": {
        "en": "Use WELCOME10 for 10% off your first order. Seasonal deals are available across categories; check Offers for current terms.",
        "kn": "ಮೊದಲ ಆರ್ಡರ್‌ಗೆ 10% ರಿಯಾಯಿತಿಗಾಗಿ WELCOME10 ಬಳಸಿ. ವಿಭಾಗಗಳಾದ್ಯಂತ ಋತುಮಾನದ ಆಫರ್‌ಗಳಿವೆ; ನಿಯಮಗಳಿಗೆ Offers ನೋಡಿ.",
        "hi": "पहले ऑर्डर पर 10% छूट के लिए WELCOME10 इस्तेमाल करें। सभी श्रेणियों में मौसमी ऑफ़र हैं; शर्तों के लिए Offers देखें।",
        "ta": "முதல் ஆர்டருக்கு 10% தள்ளுபடி பெற WELCOME10 பயன்படுத்துங்கள். பருவகால சலுகைகளுக்கு Offers பகுதியைப் பார்க்கவும்.",
        "te": "మొదటి ఆర్డర్‌పై 10% తగ్గింపుకు WELCOME10 ఉపయోగించండి. అన్ని విభాగాల్లో సీజనల్ ఆఫర్లు ఉన్నాయి; నిబంధనలకు Offers చూడండి.",
    },
    "catalog": {
        "en": "The catalog has 20 products across Electronics, Fashion & Apparel, Home & Kitchen, Beauty & Personal Care, Sports & Fitness, and Books.",
        "kn": "ಕ್ಯಾಟಲಾಗ್‌ನಲ್ಲಿ Electronics, Fashion & Apparel, Home & Kitchen, Beauty & Personal Care, Sports & Fitness ಮತ್ತು Books ವಿಭಾಗಗಳ 20 ಉತ್ಪನ್ನಗಳಿವೆ.",
        "hi": "कैटलॉग में Electronics, Fashion & Apparel, Home & Kitchen, Beauty & Personal Care, Sports & Fitness और Books के 20 उत्पाद हैं।",
        "ta": "Electronics, Fashion & Apparel, Home & Kitchen, Beauty & Personal Care, Sports & Fitness, Books ஆகிய பிரிவுகளில் 20 பொருட்கள் உள்ளன.",
        "te": "కేటలాగ్‌లో Electronics, Fashion & Apparel, Home & Kitchen, Beauty & Personal Care, Sports & Fitness, Books విభాగాల్లో 20 ఉత్పత్తులు ఉన్నాయి.",
    },
    "starting_price": {
        "en": "Catalog prices start at ₹499. The lowest-priced item is The Quiet Horizon; name a product for its exact price.",
        "kn": "ಕ್ಯಾಟಲಾಗ್ ಬೆಲೆಗಳು ₹499 ರಿಂದ ಪ್ರಾರಂಭವಾಗುತ್ತವೆ. ಕಡಿಮೆ ಬೆಲೆಯ ಉತ್ಪನ್ನ The Quiet Horizon; ನಿಖರ ಬೆಲೆಗೆ ಉತ್ಪನ್ನದ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
        "hi": "कैटलॉग की कीमतें ₹499 से शुरू होती हैं। सबसे कम कीमत वाला उत्पाद The Quiet Horizon है; सटीक कीमत के लिए उत्पाद का नाम बताइए।",
        "ta": "பட்டியல் விலைகள் ₹499 முதல் தொடங்குகின்றன. குறைந்த விலை The Quiet Horizon; துல்லியமான விலைக்கு பொருளின் பெயரைக் கூறுங்கள்.",
        "te": "కేటలాగ్ ధరలు ₹499 నుంచి ప్రారంభమవుతాయి. అతి తక్కువ ధర ఉత్పత్తి The Quiet Horizon; ఖచ్చితమైన ధరకు ఉత్పత్తి పేరు చెప్పండి.",
    },
    "recommendations": {
        "en": "Choose a product in the Product Recommender tab to see three similar items ranked by description similarity.",
        "kn": "Product Recommender ಟ್ಯಾಬ್‌ನಲ್ಲಿ ಉತ್ಪನ್ನ ಆಯ್ಕೆಮಾಡಿ; ವಿವರಣೆಯ ಹೋಲಿಕೆಯ ಆಧಾರದ ಮೇಲೆ ಮೂರು ಉತ್ಪನ್ನಗಳನ್ನು ನೋಡಬಹುದು.",
        "hi": "Product Recommender टैब में उत्पाद चुनें। विवरण की समानता के आधार पर तीन मिलते-जुलते उत्पाद दिखेंगे।",
        "ta": "Product Recommender தாவலில் ஒரு பொருளைத் தேர்ந்தெடுக்கவும். விளக்க ஒற்றுமைப்படி மூன்று பொருட்கள் காட்டப்படும்.",
        "te": "Product Recommender ట్యాబ్‌లో ఉత్పత్తిని ఎంచుకోండి. వివరణ సారూప్యత ఆధారంగా మూడు ఉత్పత్తులు చూపబడతాయి.",
    },
    "bestsellers": {
        "en": "Popular picks include the Nova X Pro Smartphone, FlexRun Running Shoes, GlowPure Vitamin C Face Serum, BrewMate French Press, PowerFlex Yoga Mat, and The Quiet Horizon.",
        "kn": "ಜನಪ್ರಿಯ ಆಯ್ಕೆಗಳು Nova X Pro Smartphone, FlexRun Running Shoes, GlowPure Vitamin C Face Serum, BrewMate French Press, PowerFlex Yoga Mat ಮತ್ತು The Quiet Horizon.",
        "hi": "लोकप्रिय विकल्पों में Nova X Pro Smartphone, FlexRun Running Shoes, GlowPure Vitamin C Face Serum, BrewMate French Press, PowerFlex Yoga Mat और The Quiet Horizon शामिल हैं।",
        "ta": "பிரபலமான தேர்வுகள் Nova X Pro Smartphone, FlexRun Running Shoes, GlowPure Vitamin C Face Serum, BrewMate French Press, PowerFlex Yoga Mat, The Quiet Horizon.",
        "te": "ప్రజాదరణ పొందినవి Nova X Pro Smartphone, FlexRun Running Shoes, GlowPure Vitamin C Face Serum, BrewMate French Press, PowerFlex Yoga Mat, The Quiet Horizon.",
    },
    "warranty": {
        "en": "Electronics include a 1-year manufacturer warranty. Check the product listing for the exact coverage and exclusions.",
        "kn": "ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್‌ಗೆ ತಯಾರಕರ 1 ವರ್ಷದ ವಾರಂಟಿ ಇದೆ. ನಿಖರ ವ್ಯಾಪ್ತಿ ಮತ್ತು ಹೊರತಾಗುವಿಕೆಗಳಿಗೆ ಉತ್ಪನ್ನ ಪಟ್ಟಿ ನೋಡಿ.",
        "hi": "इलेक्ट्रॉनिक्स पर निर्माता की 1 साल की वारंटी है। कवरेज और अपवादों के लिए उत्पाद सूची देखें।",
        "ta": "எலக்ட்ரானிக்ஸ் பொருட்களுக்கு உற்பத்தியாளரின் 1 ஆண்டு உத்தரவாதம் உண்டு. விதிவிலக்குகளுக்கு தயாரிப்பு பட்டியலைப் பார்க்கவும்.",
        "te": "ఎలక్ట్రానిక్స్‌కు తయారీదారు 1 సంవత్సరం వారంటీ ఉంటుంది. కవరేజ్, మినహాయింపులకు ఉత్పత్తి జాబితా చూడండి.",
    },
    "damaged": {
        "en": "Report a damaged or defective delivery in My Orders within 7 days. Eligible items receive a replacement after verification.",
        "kn": "ಹಾನಿಗೊಂಡ ಅಥವಾ ದೋಷಯುಕ್ತ ವಸ್ತುವನ್ನು 7 ದಿನಗಳೊಳಗೆ My Orders ನಲ್ಲಿ ವರದಿ ಮಾಡಿ. ಪರಿಶೀಲನೆಯ ನಂತರ ಅರ್ಹ ವಸ್ತು ಬದಲಾಯಿಸಲಾಗುತ್ತದೆ.",
        "hi": "क्षतिग्रस्त या खराब वस्तु की सूचना 7 दिनों में My Orders में दें। जाँच के बाद पात्र वस्तु बदली जाएगी।",
        "ta": "சேதமடைந்த அல்லது பழுதான பொருளை 7 நாட்களுக்குள் My Orders-இல் தெரிவிக்கவும். சரிபார்த்த பின் தகுதியுள்ள பொருள் மாற்றப்படும்.",
        "te": "దెబ్బతిన్న లేదా లోపభూయిష్ట వస్తువును 7 రోజుల్లో My Ordersలో నివేదించండి. తనిఖీ తర్వాత అర్హమైన వస్తువును మారుస్తాం.",
    },
    "support": {
        "en": "24/7 AI voice and text chat is available here. Call toll-free 1800-123-4567 or email support@smartecommerce.ai for customer support.",
        "kn": "24/7 AI ಧ್ವನಿ ಮತ್ತು ಪಠ್ಯ ಚಾಟ್ ಇಲ್ಲಿ ಲಭ್ಯವಿದೆ. ಗ್ರಾಹಕ ಸಹಾಯಕ್ಕೆ ಟೋಲ್-ಫ್ರೀ 1800-123-4567 ಕರೆಮಾಡಿ ಅಥವಾ support@smartecommerce.ai ಗೆ ಇಮೇಲ್ ಮಾಡಿ.",
        "hi": "24/7 AI वॉइस और टेक्स्ट चैट यहाँ उपलब्ध है। सहायता के लिए टोल-फ्री 1800-123-4567 पर कॉल करें या support@smartecommerce.ai पर ईमेल करें।",
        "ta": "24/7 AI குரல் மற்றும் உரை அரட்டை இங்கே கிடைக்கும். உதவிக்கு கட்டணமில்லா 1800-123-4567 அல்லது support@smartecommerce.ai-ஐ தொடர்புகொள்ளுங்கள்.",
        "te": "24/7 AI వాయిస్, టెక్స్ట్ చాట్ ఇక్కడ అందుబాటులో ఉన్నాయి. సహాయం కోసం టోల్-ఫ్రీ 1800-123-4567కు కాల్ చేయండి లేదా support@smartecommerce.aiకు ఇమెయిల్ చేయండి.",
    },
    "greeting": {
        "en": "Hello! Ask me about delivery, returns, payments, orders, offers, products, warranty, or support.",
        "kn": "ನಮಸ್ಕಾರ! ಡೆಲಿವರಿ, ರಿಟರ್ನ್, ಪಾವತಿ, ಆರ್ಡರ್, ಆಫರ್, ಉತ್ಪನ್ನ, ವಾರಂಟಿ ಅಥವಾ ಸಹಾಯದ ಬಗ್ಗೆ ಕೇಳಿ.",
        "hi": "नमस्ते! डिलीवरी, रिटर्न, भुगतान, ऑर्डर, ऑफ़र, उत्पाद, वारंटी या सहायता के बारे में पूछें।",
        "ta": "வணக்கம்! விநியோகம், திரும்பப்பெறுதல், கட்டணம், ஆர்டர், சலுகை, பொருள், உத்தரவாதம் அல்லது உதவி பற்றி கேளுங்கள்.",
        "te": "నమస్కారం! డెలివరీ, రిటర్న్, చెల్లింపులు, ఆర్డర్లు, ఆఫర్లు, ఉత్పత్తులు, వారంటీ లేదా సహాయం గురించి అడగండి.",
    },
    "thanks": {
        "en": "You're welcome. I can help with delivery, returns, payments, order tracking, offers, products, warranty, and support.",
        "kn": "ಸ್ವಾಗತ. ಡೆಲಿವರಿ, ರಿಟರ್ನ್, ಪಾವತಿ, ಆರ್ಡರ್ ಟ್ರ್ಯಾಕಿಂಗ್, ಆಫರ್, ಉತ್ಪನ್ನ, ವಾರಂಟಿ ಮತ್ತು ಸಹಾಯಕ್ಕೆ ನಾನು ನೆರವಾಗುತ್ತೇನೆ.",
        "hi": "आपका स्वागत है। मैं डिलीवरी, रिटर्न, भुगतान, ऑर्डर ट्रैकिंग, ऑफ़र, उत्पाद, वारंटी और सहायता में मदद कर सकता हूँ।",
        "ta": "வரவேற்கிறோம். விநியோகம், திரும்பப்பெறுதல், கட்டணம், ஆர்டர் கண்காணிப்பு, சலுகை, பொருள், உத்தரவாதம், உதவிக்கு நான் உதவுவேன்.",
        "te": "స్వాగతం. డెలివరీ, రిటర్న్, చెల్లింపులు, ఆర్డర్ ట్రాకింగ్, ఆఫర్లు, ఉత్పత్తులు, వారంటీ, సపోర్ట్‌లో సహాయం చేస్తాను.",
    },
    "app": {
        "en": "This app combines product recommendations, a multilingual voice/text shopping assistant, and a sample-based fake-review detector.",
        "kn": "ಈ ಆ್ಯಪ್ ಉತ್ಪನ್ನ ಶಿಫಾರಸು, ಬಹುಭಾಷಾ ಧ್ವನಿ/ಪಠ್ಯ ಸಹಾಯಕ ಮತ್ತು ಮಾದರಿ ಆಧಾರಿತ ನಕಲಿ ವಿಮರ್ಶೆ ಪತ್ತೆಹಚ್ಚುವಿಕೆಯನ್ನು ಒಳಗೊಂಡಿದೆ.",
        "hi": "इस ऐप में उत्पाद सुझाव, बहुभाषी वॉइस/टेक्स्ट सहायक और नमूना-आधारित नकली समीक्षा पहचान शामिल है।",
        "ta": "இந்த செயலியில் தயாரிப்பு பரிந்துரை, பன்மொழி குரல்/உரை உதவியாளர், மாதிரி அடிப்படையிலான போலி மதிப்புரை கண்டறிதல் உள்ளன.",
        "te": "ఈ యాప్‌లో ఉత్పత్తి సిఫార్సులు, బహుభాషా వాయిస్/టెక్స్ట్ సహాయకుడు, నమూనా ఆధారిత నకిలీ సమీక్ష గుర్తింపు ఉన్నాయి.",
    },
    "reviews": {
        "en": "Paste a review in the Fake Review Detector tab. Its sample-trained model estimates whether the text looks genuine or suspicious; it is not definitive.",
        "kn": "Fake Review Detector ಟ್ಯಾಬ್‌ನಲ್ಲಿ ವಿಮರ್ಶೆ ಅಂಟಿಸಿ. ಮಾದರಿ ತರಬೇತಿ ಪಡೆದ ಮಾದರಿ ನಿಜವಾದ ಅಥವಾ ಅನುಮಾನಾಸ್ಪದ ಎಂದು ಅಂದಾಜಿಸುತ್ತದೆ; ಇದು ಅಂತಿಮ ತೀರ್ಪಲ್ಲ.",
        "hi": "Fake Review Detector टैब में समीक्षा डालें। नमूना-प्रशिक्षित मॉडल उसके वास्तविक या संदिग्ध होने का अनुमान देता है, अंतिम निर्णय नहीं।",
        "ta": "Fake Review Detector தாவலில் மதிப்புரையை ஒட்டவும். மாதிரி பயிற்சி பெற்ற மாடல் உண்மையா சந்தேகத்திற்குரியதா என மதிப்பிடும்; இது இறுதி முடிவு அல்ல.",
        "te": "Fake Review Detector ట్యాబ్‌లో సమీక్షను అతికించండి. నమూనాతో శిక్షణ పొందిన మోడల్ నిజమైనదా అనుమానాస్పదమా అని అంచనా వేస్తుంది; ఇది తుది తీర్పు కాదు.",
    },
}

FAQ_KNOWLEDGE_BASE = [
    {"answer": "services", "questions": ("what services do you provide", "what can this store do", "ನಿಮ್ಮ ಸೇವೆಗಳು ಯಾವುವು", "आपकी सेवाएं क्या हैं", "உங்கள் சேவைகள் என்ன", "మీ సేవలు ఏమిటి"), "keywords": ("services", "features", "ಸೇವೆಗಳು", "सेवाएं", "சேவைகள்", "సేవలు")},
    {"answer": "delivery", "questions": ("how long does delivery take", "shipping time", "ಡೆಲಿವರಿ ಎಷ್ಟು ದಿನ", "ಡெಲಿವರಿ ಎಷ್ಟು ದಿನ", "डिलीवरी में कितना समय", "डिलीवरी कितने दिन में होगी", "டெலிவரி எத்தனை நாள்", "டெலிவரி எத்தனை நாட்கள்", "డెలివరీ ఎన్ని రోజులు", "డెలివరీ ఎన్ని రోజుల్లో"), "keywords": ("delivery", "shipping time", "ಎಷ್ಟು ದಿನ", "ಕितने दिन", "डिलीवरी", "कितना समय", "எத்தனை நாள்", "டெலிவரி", "ఎన్ని రోజులు", "డెలివరీ")},
    {"answer": "shipping_fee", "questions": ("is shipping free", "free delivery above what amount", "ಉಚಿತ ಡೆಲಿವರಿ ಎಷ್ಟು ರೂಪಾಯಿಗೆ", "मुफ्त डिलीवरी कितने पर", "இலவச விநியோகம் எவ்வளவு", "ఉచిత డెలివరీ ఎంతకు"), "keywords": ("shipping fee", "free delivery", "ಉಚಿತ ಡೆಲಿವರಿ", "मुफ्त डिलीवरी", "இலவச விநியோகம்", "ఉచిత డెలివరీ")},
    {"answer": "delivery", "questions": ("do you deliver to Mysore", "shipping to Bengaluru Karnataka", "ಮೈಸೂರಿಗೆ ಡೆಲಿವರಿ ಇದೆಯೇ", "क्या मैसूर में डिलीवरी है", "பெங்களூருக்கு அனுப்புவீர்களா", "మైసూరులో డెలివరీ ఉందా"), "keywords": ("mysore", "mysuru", "ಮೈಸೂರು", "मैसूर", "ಮೈಸೂರು", "मैसூர்", "பெங்களூர்", "బెంగళూరు")},
    {"answer": "returns", "questions": ("what is your return policy", "can I return within seven days", "ರಿಟರ್ನ್ ಪಾಲಿಸಿ ಏನು", "रिटर्न पॉलिसी क्या है", "ரிட்டர்ன் விதிமுறை என்ன", "రిటర్న్ పాలసీ ఏమిటి"), "keywords": ("returns", "return policy", "ರಿಟರ್ನ್", "ಹಿಂತಿರುಗಿಸಿ", "रिटर्न", "रिटर्न")},
    {"answer": "returns", "questions": ("are returns no questions asked", "easy returns", "ಯಾವುದೇ ಪ್ರಶ್ನೆ ಇಲ್ಲದೆ ವಾಪಸ್ ಮಾಡಬಹುದೇ", "बिना सवाल वापसी", "கேள்வியின்றி திருப்பலாமா", "ప్రశ్నలు లేకుండా తిరిగి ఇవ్వవచ్చా"), "keywords": ("no questions asked", "easy return", "ಯಾವುದೇ ಪ್ರಶ್ನೆ", "बिना सवाल", "கேள்வியின்றி", "ప్రశ్నలు లేకుండా")},
    {"answer": "returns", "questions": ("how do I schedule a return pickup", "instant pickup for returns", "ರಿಟರ್ನ್ ಪಿಕಪ್ ಹೇಗೆ", "रिटर्न पिकअप कैसे करें", "திரும்பப் பெற பிக்கப் எப்படி", "రిటర్న్ పికప్ ఎలా"), "keywords": ("pickup", "pick up", "ಪಿಕಪ್", "पिकअप", "பிக்கப்", "పికప్")},
    {"answer": "refund", "questions": ("when will my refund arrive", "refund processing time", "ರಿಫಂಡ್ ಯಾವಾಗ ಬರುತ್ತದೆ", "रिफंड कब मिलेगा", "பணம் எப்போது திரும்பும்", "రీఫండ్ ఎప్పుడు వస్తుంది"), "keywords": ("refund", "money back", "ರಿಫಂಡ್", "ಮರುಪಾವತಿ", "रिफंड", "ரீஃபண்ட்", "రీఫండ్")},
    {"answer": "exchange", "questions": ("can I exchange an item", "how do exchanges work", "ವಸ್ತು ಬದಲಾಯಿಸಬಹುದೇ", "क्या सामान बदल सकते हैं", "பொருளை மாற்ற முடியுமா", "వస్తువును మార్చవచ్చా"), "keywords": ("exchange", "swap", "ಬದಲಾಯಿಸಿ", "ಬದಲಾವಣೆ", "एक्सचेंज", "மாற்ற", "మార్పిడి")},
    {"answer": "payments", "questions": ("what payment methods do you accept", "how can I pay", "ಯಾವ ಪಾವತಿ ವಿಧಾನಗಳು", "कौन से पेमेंट तरीके", "என்ன கட்டண முறைகள்", "ఏ చెల్లింపు పద్ధతులు"), "keywords": ("payment methods", "pay", "ಪೇಮೆಂಟ್", "ಪಾವತಿ", "पेमेंट", "भुगतान", "கட்டணம்", "చెల్లింపు")},
    {"answer": "payments", "questions": ("can I pay using UPI GPay PhonePe or Paytm", "do you accept cards and net banking", "UPI GPay PhonePe ಸ್ವೀಕರಿಸುತ್ತೀರಾ", "क्या GPay PhonePe लेते हैं", "GPay PhonePe ஏற்கிறீர்களா", "GPay PhonePe తీసుకుంటారా"), "keywords": ("upi", "gpay", "google pay", "phonepe", "paytm", "card", "net banking", "ಜಿಪೇ", "ಯುಪಿಐ", "फोनपे", "पेमेंट", "யுபிஐ", "ఫోన్ పే")},
    {"answer": "payments", "questions": ("is cash on delivery available", "can I pay cash when it arrives", "ಕ್ಯಾಶ್ ಆನ್ ಡೆಲಿವರಿ ಇದೆಯೇ", "कैश ऑन डिलीवरी है", "பணம் செலுத்தி பெறலாமா", "క్యాష్ ఆన్ డెలివరీ ఉందా"), "keywords": ("cash on delivery", "cod", "pay cash", "ಕ್ಯಾಶ್ ಆನ್ ಡೆಲಿವರಿ", "नकद", "பணம் செலுத்தி", "క్యాష్")},
    {"answer": "tracking", "questions": ("how can I track my order", "where is my order", "ನನ್ನ ಆರ್ಡರ್ ಎಲ್ಲಿ", "मेरा ऑर्डर कहां है", "என் ஆர்டர் எங்கே", "నా ఆర్డర్ ఎక్కడ"), "keywords": ("track order", "order status", "order id", "ಆರ್ಡರ್ ಟ್ರ್ಯಾಕ್", "ಆರ್ಡರ್ ಎಲ್ಲಿ", "ऑर्डर ट्रैक", "ஆர்டர் எங்கே", "ఆర్డర్ ఎక్కడ")},
    {"answer": "cancel", "questions": ("can I cancel before dispatch", "how do I cancel an order", "ರವಾನೆಗೆ ಮೊದಲು ಆರ್ಡರ್ ರದ್ದು ಮಾಡಬಹುದೇ", "डिस्पैच से पहले रद्द कर सकते हैं", "அனுப்பும் முன் ரத்து செய்யலாமா", "పంపే ముందు రద్దు చేయవచ్చా"), "keywords": ("cancel", "cancellation", "dispatch", "ರದ್ದು", "ರದ್ದುಮಾಡಿ", "रद्द", "ரத்து", "రద్దు")},
    {"answer": "offers", "questions": ("do you have a first order discount", "WELCOME10 coupon", "ಮೊದಲ ಆರ್ಡರ್ ರಿಯಾಯಿತಿ ಇದೆಯೇ", "पहले ऑर्डर पर छूट", "முதல் ஆர்டருக்கு தள்ளுபடி", "మొదటి ఆర్డర్ తగ్గింపు"), "keywords": ("first order", "welcome10", "discount", "coupon", "ಡಿಸ್ಕೌಂಟ್", "ಆಫರ್", "छूट", "कूपन", "தள்ளுபடி", "కూపన్")},
    {"answer": "offers", "questions": ("what seasonal offers are available", "any coupons or sale", "ಋತುಮಾನದ ಆಫರ್ ಇದೆಯೇ", "सेल या कूपन है", "சலுகை அல்லது கூப்பன் உள்ளதா", "ఆఫర్లు లేదా కూపన్లు ఉన్నాయా"), "keywords": ("seasonal", "offer", "coupon", "sale", "ಆಫರ್", "கூப்பன்", "కూపన్")},
    {"answer": "catalog", "questions": ("what categories do you sell", "what is in your catalog", "ನಿಮ್ಮ ಕ್ಯಾಟಲಾಗ್‌ನಲ್ಲಿ ಏನಿದೆ", "आपके कैटलॉग में क्या है", "உங்கள் பட்டியலில் என்ன உள்ளது", "మీ కేటలాగ్‌లో ఏమున్నాయి"), "keywords": ("catalog", "categories", "products", "ಕ್ಯಾಟಲಾಗ್", "ವಿಭಾಗ", "कैटलॉग", "श्रेणियां", "பட்டியல்", "కేటలాగ్")},
    {"answer": "starting_price", "questions": ("what is the cheapest item", "what is the starting price", "ಕಡಿಮೆ ಬೆಲೆ ಎಷ್ಟು", "सबसे कम कीमत क्या है", "குறைந்த விலை என்ன", "అత్యల్ప ధర ఎంత"), "keywords": ("starting price", "cheapest", "lowest price", "ಬೆಲೆ", "ಕಡಿಮೆ ಬೆಲೆ", "कीमत", "सबसे कम", "குறைந்த விலை", "అత్యల్ప ధర")},
    {"answer": "recommendations", "questions": ("how does product recommendation work", "suggest similar products", "ಶಿಫಾರಸು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ", "उत्पाद सुझाव कैसे मिलते हैं", "பரிந்துரை எப்படி செயல்படும்", "సిఫార్సులు ఎలా పనిచేస్తాయి"), "keywords": ("recommendation", "similar products", "ಶಿಫಾರಸು", "सुझाव", "பரிந்துரை", "సిఫార్సు")},
    {"answer": "bestsellers", "questions": ("what are your best sellers", "popular products in each category", "ಜನಪ್ರಿಯ ಉತ್ಪನ್ನಗಳು ಯಾವುವು", "सबसे लोकप्रिय उत्पाद कौन से हैं", "பிரபலமான பொருட்கள் எவை", "పాపులర్ ఉత్పత్తులు ఏవి"), "keywords": ("best seller", "popular products", "top items", "ಜನಪ್ರಿಯ", "लोकप्रिय", "பிரபலமான", "ప్రజాదరణ")},
    {"answer": "warranty", "questions": ("what warranty do electronics have", "is there a one year warranty", "ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್‌ಗೆ ವಾರಂಟಿ ಇದೆಯೇ", "इलेक्ट्रॉनिक्स की वारंटी कितनी है", "எலக்ட்ரானிக்ஸ் உத்தரவாதம்", "ఎలక్ట్రానిక్స్ వారంటీ ఎంత"), "keywords": ("warranty", "guarantee", "ವಾರಂಟಿ", "ಖಾತರಿ", "वारंटी", "உத்தரவாதம்", "వారంటీ")},
    {"answer": "damaged", "questions": ("my item arrived damaged", "how do I replace a defective product", "ಹಾನಿಗೊಂಡ ವಸ್ತು ಬದಲಾಯಿಸಿ", "खराब सामान बदलना है", "சேதமான பொருளை மாற்ற வேண்டும்", "దెబ్బతిన్న వస్తువు మార్చాలి"), "keywords": ("damaged", "defective", "replacement", "broken", "ಹಾನಿಗೊಂಡ", "ದೋಷ", "खराब", "क्षतिग्रस्त", "சேதம்", "దెబ్బతిన్న")},
    {"answer": "support", "questions": ("how do I contact customer support", "customer care phone and email", "ಗ್ರಾಹಕ ಸೇವೆಯನ್ನು ಹೇಗೆ ಸಂಪರ್ಕಿಸಲಿ", "ग्राहक सेवा से कैसे संपर्क करें", "வாடிக்கையாளர் சேவையை எப்படி தொடர்பு கொள்வது", "కస్టమర్ కేర్‌ను ఎలా సంప్రదించాలి"), "keywords": ("customer support", "customer care", "contact", "helpline", "ಗ್ರಾಹಕ ಸೇವೆ", "ಸಂಪರ್ಕ", "ग्राहक सेवा", "संपर्क", "வாடிக்கையாளர் சேவை", "సంప్రదించండి")},
    {"answer": "support", "questions": ("is chat available 24 hours", "are you open 24/7", "24/7 ಸಹಾಯ ಇದೆಯೇ", "क्या चैट 24 घंटे उपलब्ध है", "24 மணி நேர ஆதரவு உள்ளதா", "24 గంటల సహాయం ఉందా"), "keywords": ("24/7", "24 hours", "always available", "24 ಗಂಟೆ", "24 घंटे", "24 மணி", "24 గంటలు")},
    {"answer": "greeting", "questions": ("hello", "good morning", "ನಮಸ್ಕಾರ", "नमस्ते", "வணக்கம்", "నమస్కారం"), "keywords": ("hello", "hi", "hey", "ನಮಸ್ಕಾರ", "ನಮಸ್ತೆ", "नमस्ते", "வணக்கம்", "హలో")},
    {"answer": "thanks", "questions": ("thank you", "thanks for your help", "ಧನ್ಯವಾದಗಳು", "धन्यवाद", "நன்றி", "ధన్యవాదాలు"), "keywords": ("thanks", "thank you", "ಧನ್ಯವಾದ", "ಧನ್ಯವಾದಗಳು", "धन्यवाद", "நன்றி", "ధన్యవాదాలు")},
    {"answer": "app", "questions": ("what is this app", "what features does this app have", "ಈ ಆ್ಯಪ್ ಏನು ಮಾಡುತ್ತದೆ", "यह ऐप क्या करता है", "இந்த செயலி என்ன செய்கிறது", "ఈ యాప్ ఏమి చేస్తుంది"), "keywords": ("app", "application", "features", "ಆ್ಯಪ್", "ಸೌಲಭ್ಯ", "ऐप", "सुविधाएं", "செயலி", "యాప్")},
    {"answer": "reviews", "questions": ("how do I check a fake review", "what does the review detector do", "ನಕಲಿ ವಿಮರ್ಶೆ ಹೇಗೆ ಪರಿಶೀಲಿಸಲಿ", "फर्जी समीक्षा कैसे जांचें", "போலி மதிப்புரையை எப்படி சரிபார்ப்பது", "నకిలీ సమీక్షను ఎలా తనిఖీ చేయాలి"), "keywords": ("fake review", "review detector", "ನಕಲಿ ವಿಮರ್ಶೆ", "फर्जी समीक्षा", "போலி மதிப்புரை", "నకిలీ సమీక్ష")},
    {"answer": "catalog", "questions": ("do you sell electronics and fashion", "do you have books beauty and sports items", "ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್ ಮತ್ತು ಫ್ಯಾಷನ್ ಇದೆಯೇ", "क्या किताबें और ब्यूटी उत्पाद हैं", "புத்தகங்கள் அழகு பொருட்கள் உள்ளனவா", "పుస్తకాలు బ్యూటీ వస్తువులు ఉన్నాయా"), "keywords": ("electronics", "fashion", "beauty", "books", "sports", "ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್", "ಫ್ಯಾಷನ್", "ಪುಸ್ತಕ", "किताब", "புத்தகம்", "పుస్తకాలు")},
    {"answer": "tracking", "questions": ("where can I find tracking number", "track using order ID", "Order ID ಮೂಲಕ ಟ್ರ್ಯಾಕ್ ಮಾಡುವುದು ಹೇಗೆ", "Order ID से ट्रैक कैसे करें", "Order ID மூலம் கண்காணிப்பது எப்படி", "Order IDతో ఎలా ట్రాక్ చేయాలి"), "keywords": ("tracking number", "order id", "track", "ಟ್ರ್ಯಾಕ್", "ऑर्डर ID", "கண்காணிப்பு", "ట్రాక్")},
    {"answer": "refund", "questions": ("how many days for refund credit", "refund to original payment method", "ಮೂಲ ಪಾವತಿಗೆ ರಿಫಂಡ್ ಯಾವಾಗ", "मूल भुगतान में रिफंड कब", "அசல் கட்டணத்திற்கு பணம் எப்போது", "అసలు చెల్లింపుకు రీఫండ్ ఎప్పుడు"), "keywords": ("refund credited", "business days", "ಮೂಲ ಪಾವತಿ", "ರಿಫಂಡ್", "रिफंड", "மூல கட்டணம்", "రీఫండ్")},
    {"answer": "offers", "questions": ("WELCOME10 gives how much off", "first purchase promo code", "WELCOME10 ಎಷ್ಟು ರಿಯಾಯಿತಿ", "WELCOME10 छूट कितनी", "WELCOME10 தள்ளுபடி எவ்வளவு", "WELCOME10 తగ్గింపు ఎంత"), "keywords": ("welcome10", "first order", "10% off", "coupon", "ರಿಯಾಯಿತಿ", "छूट", "தள்ளுபடி", "తగ్గింపు")},
    {"answer": "recommendations", "questions": ("how many similar products are recommended", "where is the recommender tab", "ಎಷ್ಟು ಹೋಲುವ ಉತ್ಪನ್ನಗಳು", "कितने समान उत्पाद दिखते हैं", "எத்தனை ஒத்த பொருட்கள்", "ఎన్ని సారూప్య ఉత్పత్తులు"), "keywords": ("three recommendations", "similar products", "recommender tab", "ಹೋಲುವ ಉತ್ಪನ್ನ", "समान उत्पाद", "ஒத்த பொருட்கள்", "సారూప్య")},
    {"answer": "damaged", "questions": ("replacement for damaged item in seven days", "item is broken what should I do", "7 ದಿನಗಳಲ್ಲಿ ಹಾನಿಗೊಂಡ ವಸ್ತು ಬದಲಾವಣೆ", "7 दिन में खराब सामान बदलना", "7 நாட்களில் சேதமான பொருள் மாற்றம்", "7 రోజుల్లో దెబ్బతిన్న వస్తువు మార్పిడి"), "keywords": ("damaged replacement", "seven day replacement", "ಹಾನಿಗೊಂಡ", "ಬದಲಾವಣೆ", "खराब सामान", "சேதமான", "దెబ్బతిన్న")},
    {"answer": "support", "questions": ("what is the toll free number", "what is support email", "ಟೋಲ್ ಫ್ರೀ ಸಂಖ್ಯೆ ಏನು", "टोल फ्री नंबर क्या है", "கட்டணமில்லா எண் என்ன", "టోల్ ఫ్రీ నంబర్ ఏమిటి"), "keywords": ("1800-123-4567", "support@smartecommerce.ai", "toll free", "ಟೋಲ್ ಫ್ರೀ", "टोल फ्री", "கட்டணமில்லா", "టోల్ ఫ్రీ")},
    {"answer": "payments", "questions": ("is payment secure", "are card payments safe", "ಪಾವತಿ ಸುರಕ್ಷಿತವೇ", "क्या भुगतान सुरक्षित है", "கட்டணம் பாதுகாப்பானதா", "చెల్లింపు సురక్షితమేనా"), "keywords": ("secure payment", "safe payment", "ಸುರಕ್ಷಿತ ಪಾವತಿ", "सुरक्षित भुगतान", "பாதுகாப்பான கட்டணம்", "సురక్షిత చెల్లింపు")},
    {"answer": "delivery", "questions": ("do you ship all over India", "how fast is delivery across India", "ಭಾರತದಾದ್ಯಂತ ಶಿಪ್ಪಿಂಗ್ ಇದೆಯೇ", "क्या पूरे भारत में शिपिंग है", "இந்தியா முழுவதும் விநியோகம் உள்ளதா", "భారతదేశం అంతటా డెలివరీ ఉందా"), "keywords": ("all India", "across India", "भारत भर", "ದೇಶಾದ್ಯಂತ", "இந்தியா முழுவதும்", "భారతదేశం అంతటా")},
    {"answer": "returns", "questions": ("do you offer seven day easy returns", "is there a return window", "7 ದಿನದ ರಿಟರ್ನ್ ಇದೆಯೇ", "7 दिन की वापसी है", "7 நாள் திரும்பப்பெறுதல் உள்ளதா", "7 రోజుల రిటర్న్ ఉందా"), "keywords": ("7 day return", "return window", "ಏಳು ದಿನ", "7 ದಿನ", "7 दिन", "7 நாள்", "7 రోజులు")},
    {"answer": "cancel", "questions": ("is cancellation free before shipping", "cancel order before dispatch", "ರವಾನೆಗೂ ಮೊದಲು ರದ್ದು ಉಚಿತವೇ", "भेजने से पहले रद्द करना मुफ्त है", "அனுப்பும் முன் ரத்து இலவசமா", "పంపే ముందు రద్దు ఉచితమా"), "keywords": ("free cancellation", "before dispatch", "ರದ್ದು", "ರವಾನೆ", "रद्द", "அனுப்பும் முன்", "రద్దు")},
    {"answer": "bestsellers", "questions": ("best product in electronics category", "popular picks for books and fitness", "ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್‌ನಲ್ಲಿ ಜನಪ್ರಿಯ ಯಾವುದು", "इलेक्ट्रॉनिक्स में लोकप्रिय क्या है", "எலக்ட்ரானிக்ஸில் பிரபலமானது", "ఎలక్ట్రానిక్స్‌లో పాపులర్ ఏది"), "keywords": ("best product", "popular", "best seller", "ಜನಪ್ರಿಯ", "लोकप्रिय", "பிரபலமான", "పాపులర్")},
    {"answer": "greeting", "questions": ("can you help me shop", "I need shopping help", "ಶಾಪಿಂಗ್ ಸಹಾಯ ಮಾಡುತ್ತೀರಾ", "खरीदारी में मदद करेंगे", "ஷாப்பிங் உதவி செய்ய முடியுமா", "షాపింగ్‌లో సహాయం చేస్తారా"), "keywords": ("help me", "shopping help", "ಸಹಾಯ", "मदद", "உதவி", "సహాయం")},
    {"answer": "services", "questions": ("do you have an AI voice assistant", "can I speak to the shopping assistant", "AI ಧ್ವನಿ ಸಹಾಯಕ ಇದೆಯೇ", "क्या AI वॉइस सहायक है", "AI குரல் உதவியாளர் உள்ளதா", "AI వాయిస్ అసిస్టెంట్ ఉందా"), "keywords": ("voice assistant", "AI assistant", "ಧ್ವನಿ ಸಹಾಯಕ", "वॉइस सहायक", "குரல் உதவியாளர்", "వాయిస్ అసిస్టెంట్")},
    {"answer": "catalog", "questions": ("are there twenty products", "how many products are in the store", "ಅಂಗಡಿಯಲ್ಲಿ ಎಷ್ಟು ಉತ್ಪನ್ನಗಳಿವೆ", "स्टोर में कितने उत्पाद हैं", "கடையில் எத்தனை பொருட்கள்", "స్టోర్‌లో ఎన్ని ఉత్పత్తులు"), "keywords": ("20 products", "how many products", "ಎಷ್ಟು ಉತ್ಪನ್ನ", "कितने उत्पाद", "எத்தனை பொருட்கள்", "ఎన్ని ఉత్పత్తులు")},
    {"answer": "starting_price", "questions": ("is there anything under five hundred rupees", "what is the minimum product price", "₹500 ಕ್ಕಿಂತ ಕಡಿಮೆ ವಸ್ತು ಇದೆಯೇ", "₹500 से कम का सामान", "₹500-க்கு குறைவான பொருள்", "₹500 కంటే తక్కువ వస్తువు"), "keywords": ("minimum price", "starting at", "cheapest", "ಕಡಿಮೆ ಬೆಲೆ", "न्यूनतम कीमत", "குறைந்த விலை", "కనిష్ట ధర")},
]

FAQ_KNOWLEDGE_BASE[1]["keywords"] += (
    "ഡെലിവറി", "കിതنے دن", "कितने दिन", "டெலிவரி", "ఎన్ని రోజులు",
)
FAQ_KNOWLEDGE_BASE[2]["keywords"] += (
    "₹499", "above ₹499", "over ₹499", "मुफ्त डिलीवरी", "ಇலவச டெலிவரி",
)

FAQ_ANSWER_INTENT = {
    "services": "services",
    "delivery": "shipping",
    "shipping_fee": "shipping_fee",
    "returns": "returns",
    "exchange": "returns",
    "refund": "returns",
    "payments": "payments",
    "tracking": "tracking",
    "cancel": "tracking",
    "offers": "offers",
    "catalog": "catalog",
    "starting_price": "catalog",
    "recommendations": "catalog",
    "bestsellers": "catalog",
    "warranty": "warranty",
    "damaged": "warranty",
    "support": "support",
    "greeting": "greeting",
    "thanks": "thanks",
    "app": "services",
    "reviews": "services",
}

PRODUCT_DETAIL_TEMPLATES = {
    "en": "{name} — {category}, ₹{price:,}. {description}",
    "kn": "{name} — {category}, ₹{price:,}. {description}",
    "hi": "{name} — {category}, ₹{price:,}। {description}",
    "ta": "{name} — {category}, ₹{price:,}. {description}",
    "te": "{name} — {category}, ₹{price:,}. {description}",
}

PRODUCT_DESCRIPTION_TRANSLATIONS = {
    "kn": {
        "Nova X Pro Smartphone": "AMOLED ಪರದೆ, ವೇಗದ ಪ್ರೊಸೆಸರ್, ದೀರ್ಘ ಬ್ಯಾಟರಿ ಮತ್ತು ಹೆಚ್ಚಿನ ರೆಸಲ್ಯೂಶನ್ ಕ್ಯಾಮೆರಾವಿರುವ 5G ಸ್ಮಾರ್ಟ್‌ಫೋನ್.",
        "SonicBeat Wireless Headphones": "ಆಕ್ಟಿವ್ ನಾಯ್ಸ್ ಕ್ಯಾನ್ಸಲೇಶನ್, ಸಮೃದ್ಧ ಬಾಸ್ ಮತ್ತು ದೀರ್ಘ ಬ್ಯಾಟರಿಯ ವೈರ್‌ಲೆಸ್ ಓವರ್-ಇಯರ್ ಹೆಡ್‌ಫೋನ್‌ಗಳು.",
        "GlowPure Vitamin C Face Serum": "ದೈನಂದಿನ ತ್ವಚೆ ಆರೈಕೆಗೆ ತೇವಾಂಶ ನೀಡುವ, ಬೇಗ ಹೀರಿಕೊಳ್ಳುವ ವಿಟಮಿನ್ C ಮುಖದ ಸೀರಮ್.",
    },
    "hi": {
        "Nova X Pro Smartphone": "AMOLED स्क्रीन, तेज़ प्रोसेसर, लंबी बैटरी और उच्च-रिज़ॉल्यूशन कैमरे वाला 5G स्मार्टफोन।",
        "SonicBeat Wireless Headphones": "एक्टिव नॉइज़ कैंसलेशन, दमदार बास और लंबी बैटरी वाले वायरलेस ओवर-ईयर हेडफोन।",
        "GlowPure Vitamin C Face Serum": "रोज़ाना त्वचा देखभाल के लिए हल्का, नमी देने वाला और जल्दी अवशोषित होने वाला विटामिन C सीरम।",
    },
    "ta": {
        "Nova X Pro Smartphone": "AMOLED திரை, வேகமான செயலி, நீண்ட பேட்டரி மற்றும் உயர் தெளிவுத்திறன் கேமரா கொண்ட 5G ஸ்மார்ட்போன்.",
        "SonicBeat Wireless Headphones": "சத்தத் தடுப்பு, செறிவான பாஸ், நீண்ட பேட்டரி கொண்ட வயர்லெஸ் ஓவர்-இயர் ஹெட்ஃபோன்கள்.",
        "GlowPure Vitamin C Face Serum": "தினசரி சருமப் பராமரிப்புக்கான ஈரப்பதமூட்டும், விரைவாக உறிஞ்சும் வைட்டமின் C சீரம்.",
    },
    "te": {
        "Nova X Pro Smartphone": "AMOLED డిస్‌ప్లే, వేగవంతమైన ప్రాసెసర్, దీర్ఘ బ్యాటరీ, అధిక రిజల్యూషన్ కెమెరా ఉన్న 5G స్మార్ట్‌ఫోన్.",
        "SonicBeat Wireless Headphones": "యాక్టివ్ నాయిస్ క్యాన్సలేషన్, బలమైన బాస్, దీర్ఘ బ్యాటరీతో వైర్‌లెస్ ఓవర్-ఇయర్ హెడ్‌ఫోన్లు.",
        "GlowPure Vitamin C Face Serum": "రోజువారీ చర్మ సంరక్షణ కోసం తేమనిచ్చే, త్వరగా శోషించబడే విటమిన్ C సీరమ్.",
    },
}

PRODUCT_QUERY_ALIASES = {
    "Nova X Pro Smartphone": ("nova x pro", "nova smartphone", "smartphone", "mobile phone", "5g phone", "ಫೋನ್", "मोबाइल", "மொபைல்", "మొబైల్"),
    "PixelView 4K Smart TV": ("pixelview", "4k smart tv", "smart tv", "television", "ಟಿವಿ", "टीवी", "தொலைக்காட்சி", "టీవీ"),
    "SonicBeat Wireless Headphones": ("sonicbeat", "headphones", "wireless headphones", "earphones", "ಹೆಡ್‌ಫೋನ್", "हेडफोन", "ஹெட்ஃபோன்", "హెడ్‌ఫోన్స్"),
    "ThunderMax Bluetooth Speaker": ("thundermax", "bluetooth speaker", "speaker", "ಸ್ಪೀಕರ್", "स्पीकर", "ஸ்பீக்கர்", "స్పీకర్"),
    "Everyday Cotton Kurta": ("cotton kurta", "kurta", "ಕುರ್ತಾ", "कुर्ता", "குர்தா", "కుర్తా"),
    "Classic Denim Jacket": ("denim jacket", "jacket", "ಡೆನಿಮ್ ಜಾಕೆಟ್", "जैकेट", "ஜாக்கெட்", "జాకెట్"),
    "FlexRun Running Shoes": ("flexrun", "running shoes", "trainers", "ರನ್ನಿಂಗ್ ಶೂ", "रनिंग शू", "ஓடும் காலணி", "రన్నింగ్ షూస్"),
    "Everyday Canvas Backpack": ("canvas backpack", "backpack", "rucksack", "ಬ್ಯಾಕ್‌ಪ್ಯಾಕ್", "बैग", "பேக்", "బ్యాక్‌ప్యాక్"),
    "BrewMate French Press": ("brewmate", "french press", "coffee maker", "ಫ್ರೆಂಚ್ ಪ್ರೆಸ್", "कॉफी मेकर", "காபி மேக்கர்", "కాఫీ మేకర్"),
    "ChefPro Nonstick Cookware Set": ("chefpro", "nonstick cookware", "cookware set", "cookware", "ನಾನ್‌ಸ್ಟಿಕ್ ಕುಕ್‌ವೇರ್", "कुकवेयर", "சமையல் பாத்திரம்", "కుక్‌వేర్"),
    "CozyNest Cotton Bedsheet": ("cozynest", "cotton bedsheet", "bedsheet", "bed sheet", "ಬೆಡ್‌ಶೀಟ್", "बेडशीट", "படுக்கை விரிப்பு", "బెడ్‌షీట్"),
    "GlowPure Vitamin C Face Serum": ("glowpure", "vitamin c face serum", "face serum", "serum", "ಸೀರಮ್", "सीरम", "சீரம்", "సీరమ్"),
    "SilkShine Hair Care Kit": ("silkshine", "hair care kit", "shampoo", "conditioner", "ಹೇರ್ ಕೇರ್", "शैम्पू", "ஷாம்பு", "షాంపూ"),
    "FreshBloom Eau de Parfum": ("freshbloom", "eau de parfum", "perfume", "fragrance", "ಪರ್ಫ್ಯೂಮ್", "इत्र", "வாசனை திரவியம்", "పెర్ఫ్యూమ్"),
    "PulseFit Smartwatch": ("pulsefit", "smartwatch", "fitness watch", "ಸ್ಮಾರ್ಟ್‌ವಾಚ್", "स्मार्टवॉच", "ஸ்மார்ட்வாட்ச்", "స్మార్ట్‌వాచ్", "സ്മാർട്ട്‌വാച്ച്", "smart watch"),
    "PowerFlex Yoga Mat": ("powerflex", "yoga mat", "exercise mat", "ಯೋಗ ಮ್ಯಾಟ್", "योगा मैट", "யோகா மேட்", "యోగా మ్యాట్"),
    "ProGrip Adjustable Dumbbells": ("progrip", "adjustable dumbbells", "dumbbells", "weights", "ಡಂಬೆಲ್ಸ್", "डंबल", "டம்ப்பெல்ஸ்", "డంబెల్స్"),
    "The Quiet Horizon": ("quiet horizon", "fiction novel", "the novel"),
    "Everyday Personal Finance": ("personal finance", "finance book", "finance guide"),
    "The Curious Science Handbook": ("curious science", "science handbook", "science book"),
}

def _faq_text(entry):
    return " ".join(
        (*entry["questions"], *entry["keywords"])
    )


@st.cache_resource
def get_chatbot_faq_index():
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 2),
        sublinear_tf=True,
        lowercase=True,
    )
    matrix = vectorizer.fit_transform([_faq_text(entry) for entry in FAQ_KNOWLEDGE_BASE])
    return vectorizer, matrix


def _localized_faq_answer(answer_id, lang_code):
    localized = CHATBOT_ANSWER_BANK.get(answer_id, {})
    if lang_code in localized:
        return localized[lang_code]

    responses = CHATBOT_RESPONSES.get(lang_code, CHATBOT_RESPONSES["en"])
    details = DETAIL_RESPONSES.get(lang_code, DETAIL_RESPONSES["en"])
    legacy_answers = {
        "delivery": responses["shipping"],
        "returns": responses["returns"],
        "payments": PAYMENT_RESPONSES.get(lang_code, PAYMENT_RESPONSES["en"]),
        "tracking": details["tracking"],
        "offers": details["offers"],
        "warranty": WARRANTY_RESPONSES.get(lang_code, WARRANTY_RESPONSES["en"]),
        "support": details["support"],
        "greeting": responses["greeting"],
        "thanks": details["thanks"],
        "catalog": details["catalog"],
        "starting_price": PRICE_PROMPTS.get(lang_code, PRICE_PROMPTS["en"]),
        "refund": responses["returns"],
        "exchange": responses["returns"],
        "shipping_fee": responses["shipping"],
        "cancel": details["tracking"],
        "recommendations": details["catalog"],
        "bestsellers": details["catalog"],
        "damaged": WARRANTY_RESPONSES.get(lang_code, WARRANTY_RESPONSES["en"]),
        "services": responses["general"],
        "app": responses["general"],
        "reviews": responses["general"],
    }
    return legacy_answers.get(answer_id, CHATBOT_ANSWER_BANK["services"]["en"])


def answer_question(question, lang_code):
    query = " ".join(question.casefold().split())
    if not query:
        return _localized_faq_answer("services", lang_code)

    for entry in FAQ_KNOWLEDGE_BASE:
        if any(" ".join(sample.casefold().split()) == query for sample in entry["questions"]):
            return _localized_faq_answer(entry["answer"], lang_code)

    def contains_phrase(phrase):
        phrase = " ".join(phrase.casefold().split())
        if not phrase:
            return False
        if phrase.isascii() and all(char.isalnum() or char.isspace() for char in phrase):
            return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", query) is not None
        return phrase in query

    product_matches = []
    for product in PRODUCTS:
        aliases = (product["name"], *PRODUCT_QUERY_ALIASES.get(product["name"], ()))
        matched_aliases = [alias for alias in aliases if contains_phrase(alias)]
        if matched_aliases:
            product_matches.append((max(map(len, matched_aliases)), product))

    if product_matches:
        product = max(product_matches, key=lambda match: match[0])[1]
        description = PRODUCT_DESCRIPTION_TRANSLATIONS.get(lang_code, {}).get(
            product["name"],
            product["description"],
        )
        template = PRODUCT_DETAIL_TEMPLATES.get(
            lang_code,
            PRODUCT_DETAIL_TEMPLATES["en"],
        )
        return template.format(
            name=product["name"],
            category=product["category"],
            price=product["price"],
            description=description,
        )

    vectorizer, faq_matrix = get_chatbot_faq_index()
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, faq_matrix).ravel()

    for index, entry in enumerate(FAQ_KNOWLEDGE_BASE):
        answer_intent = FAQ_ANSWER_INTENT[entry["answer"]]
        language_intent = (
            None if answer_intent in ("shipping", "shipping_fee")
            else answer_intent
        )
        legacy_intent = (
            None if answer_intent in ("shipping", "shipping_fee")
            else answer_intent
        )
        extra_keywords = (
            COMMON_INTENT_KEYWORDS.get(answer_intent, ())
            + LOCAL_INTENT_KEYWORDS.get(lang_code, {}).get(language_intent, ())
            + INTENT_KEYWORDS.get(lang_code, {}).get(legacy_intent, ())
        )
        matched_keywords = sum(
            1
            for keyword in (*entry["keywords"], *extra_keywords)
            if contains_phrase(keyword)
        )
        matched_questions = sum(
            1 for sample in entry["questions"] if contains_phrase(sample)
        )
        scores[index] += min(matched_keywords, 4) * 0.18
        scores[index] += min(matched_questions, 2) * 0.35

    best_entry = FAQ_KNOWLEDGE_BASE[int(scores.argmax())]
    return _localized_faq_answer(best_entry["answer"], lang_code)


def speak(text, lang_code):
    audio = BytesIO()
    gTTS(text=text, lang=lang_code).write_to_fp(audio)
    audio.seek(0)
    return audio


st.markdown(
    """
    <style>
    :root, .stApp, [data-testid="stAppViewContainer"] {
        background: #F8FAFC !important;
        color: #0F172A !important;
    }
    h1, h2, h3, p, label, span,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stTabs"] button,
    [data-testid="stTabs"] button * {
        color: #0F172A !important;
    }
    [data-testid="stHeader"] {
        background: #F8FAFC !important;
    }
    input, textarea, [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    .hero {
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
        border-radius: 22px;
        background: #E0E7FF;
        box-shadow: 0 12px 32px rgba(86, 53, 201, 0.18);
    }
    .hero h1 { margin: 0 0 .35rem; font-size: 2.2rem; }
    .hero p { margin: 0; font-size: 1.05rem; }
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
