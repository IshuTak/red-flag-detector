# test.py

from src.models.flag_detector import FlagDetector

# Create detector instance
detector = FlagDetector(model_path='models/bert_model/saved_model/best_model.pth')

# Test messages with more examples
test_messages = [
    "You must do exactly what I say",
    "I need to know where you are at all times",
    "If you really loved me, you would do this",
    "I respect your decision and support your choice",
    "You have to tell me everything you do",
    "You're just being too sensitive about this",
    "I'll make you regret this if you don't listen",
    "Nobody will ever love you like I do",
    "I'm sorry, I shouldn't have said that. I respect your feelings",
    "Let me know when you're free to talk about this"
]
def print_analysis(result):
    """Print formatted analysis results with enhanced context"""
    print("\n" + "="*50)
    print(f"Message: {result['text']}")
    print("-"*50)
    
    # Prediction indicator
    if result['prediction'] == "Red Flag":
        print(f"🚩 Prediction: {result['prediction']} ⚠️")
    else:
        print(f"🚩 Prediction: {result['prediction']} ✅")
        
    print(f"📊 Confidence: {result['confidence']:.1f}%")
    
    # Severity with appropriate icon
    severity_icons = {
        "Critical": "⛔",
        "High": "⚠️",
        "Medium": "⚡",
        "Low": "ℹ️"
    }
    print(f"{severity_icons.get(result['severity'], 'ℹ️')} Severity: {result['severity']}")

    # Context based on severity
    if result['prediction'] == "Red Flag":
        if result['severity'] == "Critical":
            print("\n⛔ Context: CRITICAL - This message shows extremely concerning behavior that requires immediate attention")
        elif result['severity'] == "High":
            print("\n⚠️ Context: This message shows serious red flag behavior that should not be ignored")
        elif result['severity'] == "Medium":
            print("\n⚡ Context: This message shows concerning behavior that should be addressed")
        else:
            print("\n📝 Context: This message shows mild concerning behavior that should be monitored")
    else:
        print("\n✅ Context: This message shows healthy communication patterns")

    # Analysis
    print("\n📋 Analysis:")
    for reason in result['reasons']:
        print(f"  {reason}")
    
    # Pattern Details
    if result['pattern_analysis']['toxic_patterns']:
        print("\n🔍 Detected Patterns:")
        for category, patterns in result['pattern_analysis']['toxic_patterns'].items():
            category_name = category.replace('_', ' ').title()
            print(f"  • {category_name}: {', '.join(patterns)}")
    
    if 'positive_patterns' in result['pattern_analysis'] and result['pattern_analysis']['positive_patterns']:
        print("\n💚 Positive Patterns:")
        for category, patterns in result['pattern_analysis']['positive_patterns'].items():
            category_name = category.replace('_', ' ').title()
            print(f"  • {category_name}: {', '.join(patterns)}")
    
    print("="*50)
# Test each message
for message in test_messages:
    result = detector.predict(message)
    print_analysis(result)