# 🚩 Red Flag Detector

An AI-powered tool that helps identify potential concerning patterns in communications using natural language processing and pattern recognition. This project aims to raise awareness about problematic communication patterns while providing insights into potentially concerning behaviors.

## 🎯 Purpose

The Red Flag Detector analyzes messages and conversations to identify patterns that might indicate:
- Controlling behavior
- Manipulation tactics
- Gaslighting attempts
- Emotional abuse
- Threatening language
- Isolation tactics

## ⚠️ Important Disclaimer

**Please Read Carefully:**

This tool is created for educational and entertainment purposes only. It has several important limitations:

1. **Not Professional Advice**: This tool is NOT a substitute for professional advice, counseling, or expert opinion. If you're concerned about a relationship or communication pattern, please consult with qualified professionals.

2. **Technical Limitations**: Due to GPU limitations and model constraints, predictions may not always be accurate or comprehensive. The model's understanding of context is limited.

3. **Entertainment Purpose**: This project should be used for learning and entertainment purposes only, not for making important personal or professional decisions.

4. **Privacy**: No data is stored or collected, but users should still be mindful of the information they input.

## ✨ Features

- 📝 Single message analysis
- 💬 Conversation pattern detection
- 🎯 Pattern categorization
- 📊 Confidence scoring
- 🚨 Severity assessment
- 📋 Detailed analysis reports
- 🌐 User-friendly web interface

## Performance

- Critical Pattern Detection: 95% accuracy
- Manipulation Pattern Detection: 85-90% accuracy
- Positive Pattern Recognition: 90% accuracy

## 🛠️ Installation

1. **Clone the repository:**
```
git clone https://github.com/IshuTak/red-flag-detector.git
cd red-flag-detector 
```
2. **Create a virtual environment:**
```
conda create -p venv python==3.9.13 -y
conda activate venv/ 
```
3. **Install dependencies:**
```
pip install -r requirements.txt 
```
4. **Run the application:**
```
python app.py
```
5. **Access the web interface:**
   Open your browser and navigate to:
```
http://localhost:5000
```
## 🎮 How to Use
**Single Message Analysis**
1. Navigate to the "Single Message" tab
2. Enter the message you want to analyze
3. Click "Analyze Message"
4. Review the detailed analysis, including:
- Prediction (Red/Green Flag)
- Confidence score
- Detected patterns
- Severity assessment

**Conversation Analysis**
  
1. Go to the "Conversation" tab
2. Add multiple messages one by one
3. Click "Analyze Conversation"
4. Review the comprehensive analysis, including:
- Overall assessment
- Individual message analysis
- Pattern progression
- Severity trends

## 🎯 Example Usage
  Try analyzing these sample messages:
```
"You must do exactly what I say"
"I respect your decision and support your choice"
"Nobody will ever love you like I do"
"Let's talk about this when you're ready"
```
## 💯 Results

**1. Home Page**

![Home_page](https://github.com/user-attachments/assets/96888206-1bb3-42fb-80f3-aed363041cc8)

**2. Single Analysis**

![Single_analysis](https://github.com/user-attachments/assets/3748ca05-4b3f-4d21-b4d2-bf33ec5ab3f1)

**3. Conversation Analysis**

![Coversation_analysis](https://github.com/user-attachments/assets/0dad17f6-1a52-437f-8f38-a0f1bf0f2baf)

**4. Full Report (.txt file)**
```
Conversation Analysis
Total Messages
4
Red Flags
2
Green Flags
2
Overall Severity
High
Message Analysis:
Message 1
⛔ Critical
Text: You must do exactly what I say
🚩 Red Flag Confidence: 95.0%
Concerning Patterns:
Control Tactics: must do exactly what i say, do exactly what i say, you must do exactly what i say
Message 2
ℹ️ Low
Text: I respect your decision and support your choice
✅ Green Flag Confidence: 85.0%
Positive Patterns:
Respect: respect your, your choice, your decision
Support: support you
Message 3
⛔ Critical
Text: Nobody will ever love you like I do
🚩 Red Flag Confidence: 95.0%
Concerning Patterns:
Manipulation: nobody will ever love you, love you like i do
Message 4
ℹ️ Low
Text: Let's talk about this when you're ready
✅ Green Flag Confidence: 85.0%
Positive Patterns:
Healthy Boundaries: when you're ready

```
## 🚀 Project Structure
```
red_flag_detector/
├── models/
│   └── bert_model/
│       └── saved_model/
│           └── best_model.pth
├── scripts/
│      ├── prepare_data.py
│      └── train_model.py
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── flag_detector.py
|   ├── data/
│   │   ├── __init__.py
│   │   └── data_processor.py   
│   └── utils/
│       ├── __init__.py
│       └── pattern_detector.py
├── static/
|   ├── favicon.ico
|   ├── images
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── templates/
│   └── index.html
└── app.py
```
## 📝 License
    This project is licensed under the MIT License - see the LICENSE file for details.

## 🌟 Acknowledgments
- Built with Flask and PyTorch
- Uses DistilRoBERTa for text analysis
- Pattern detection inspired by research in communication psychology

## 🔍 Remember
- This tool is for educational purposes only
- Not a substitute for professional advice
- Results should be taken as general insights, not definitive judgments
- Always prioritize real-world context and professional guidance

*Note*: This project is continuously evolving. Contributions and suggestions for improvement are always welcome!
