

import re
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PatternDetector:
    def __init__(self):
        
        self.toxic_patterns = {
            'threats': [
                r'\b(regret|threat|warn|revenge)\b',
                r'sorry later', 'teach you', 'lesson',
                r'\b(suffer|hurt)\b', 'consequences',
                "you'll see what happens",
                "you will see what happens",
                "you'll pay for this",
                "you will pay for this",
                "you'll regret this",
                "you will regret this",
                "watch what happens",
                "mark my words",
                "you don't know what I'm capable of",
                "you do not know what I am capable of",
                "you're playing with fire",
                "you are playing with fire",
                "you're asking for it",
                "you are asking for it"
            ],
            
            'hate_speech': [
                r'\bhate\b', r'\bdespise\b', r'\bloathe\b',
                r'\bworthless\b', r'\buseless\b',
                r'\bdie\b', r'\bkill\b', r'\bdisappear\b',
                r'\bstupid\b', r'\bidiot\b',
                "waste of space",
                "nobody likes you",
                "everyone hates you",
                "you're nothing",
                "you're trash",
                "you don't deserve",
                "you're pathetic",
                "you are nothing",
                "you are trash",
                "you do not deserve",
                "you are pathetic",
                "you make me sick"
            ],
            
            'emotional_abuse': [
                'nobody loves', 'no one cares',
                r'\bworthless\b', r'\bburden\b',
                'never good enough', 'your fault',
                'because of you', 'make me',
                'you made me',
                "you're crazy",
                "you're too sensitive",
                "you're overreacting",
                "you're being dramatic",
                "you are crazy",
                "you are too sensitive",
                "you are overreacting",
                "you are being dramatic",
                "stop being so emotional",
                "you're mental",
                "you need help",
                "what's wrong with you",
                "you're embarrassing",
                "you're a disappointment",
                "you are mental",
                "you need help",
                "what is wrong with you",
                "you are embarrassing",
                "you are a disappointment"
            ],
            
            'manipulation': [
                'if you really', 'prove that you',
                'after all I did', r'\bungrateful\b',
                r'\bselfish\b', 'if you leave',
                'need me', 'only I', 'nobody else',
                "if you loved me you would",
                "if you cared you would",
                r"nobody\s+will\s+ever\s+love\s+you",
                r"no\s+one\s+will\s+ever\s+love\s+you",
                r"only\s+[I|i]\s+will\s+love\s+you",
                r"love\s+you\s+like\s+[I|i]\s+do",
                "look what you made me do",
                "this is your fault",
                "i do everything for you",
                "after everything i've done",
                "after everything i have done",
                "you owe me",
                "i gave up everything for you",
                "you're nothing without me",
                "you are nothing without me",
                "no one will believe you"
            ],
            'emotional_manipulation': [
                r"without\s+me\s+you",
                r"nobody\s+else\s+will",
                r"no\s+one\s+else\s+will",
                r"only\s+[I|i]\s+can",
                r"only\s+[I|i]\s+will"
                ],
            'gaslighting': [
                'imagining things', 'being dramatic',
                'overreacting', 'too sensitive',
                'never happened', 'remember wrong',
                "you're making this up",
                "you are making this up",
                "that never happened",
                "you're remembering it wrong",
                "you're confused",
                "you're just being paranoid",
                "you're overthinking",
                "you are remembering it wrong",
                "you are confused",
                "you are just being paranoid",
                "you are overthinking",
                "stop being so sensitive",
                "you're seeing things",
                "you're twisting my words",
                "you're making a big deal",
                "you are seeing things",
                "you are twisting my words",
                "you are making a big deal"
            ],
            
            'control_tactics': [
                "you can't go",
                "you're not allowed",
                "you can not go",
                "you are not allowed",
                "i forbid you",
                "you need my permission",
                "i don't want you to",
                "i do not want you to",
                "you have to ask me first",
                "i decide what you can do",
                "you're not seeing them",
                "you are not seeing them",
                "delete that person",
                "block that number",
                "who are you talking to",
                "show me your phone",
                "give me your password",
                r"you must\s+.*",  # Catches "you must do exactly what I say"
                r"must do\s+.*",
                r"have to do\s+.*",
                r"need to know where\s+.*",
                r"(must|have to|need to)\s+tell me\s+.*",
                r"(must|have to|need to)\s+let me know\s+.*",
                r"need to know\s+.*\s+at all times",
                r"have to know\s+.*\s+at all times",
                r"must know\s+.*\s+at all times",
                r"do exactly what\s+.*\s+say",
                r"do what\s+.*\s+tell you",
                r"follow\s+.*\s+orders",
                r"obey\s+.*\s+commands?"
            ],
            
            'isolation_tactics': [
                "they're not good for you",
                "they are not good for you",
                "your friends are toxic",
                "your family doesn't care",
                "they're using you",
                "they're just pretending",
                "they are using you",
                "they are just pretending",
                "they talk behind your back",
                "they're not real friends",
                "they are not real friends",
                "you only need me",
                "i'm the only one who understands",
                "i am the only one who understands",
                "stay away from them",
                "don't trust them",
                "do not trust them"
            ],
            
            'love_bombing': [
                "you're my everything",
                "i can't live without you",
                "you're my whole world",
                "you are my everything",
                "i can not live without you",
                "you are my whole world",
                "no one else matters",
                "you are the only one",
                "you're the only one",
                "we're meant to be",
                "we are meant to be",
                "soulmates forever",
                "no one will ever love you like i do",
                "you're perfect for me",
                "we're destined to be together",
                "you are perfect for me",
                "we are destined to be together"
            ],
            
            'guilt_tripping': [
                "i'll hurt myself",
                "i can't go on without you",
                "you're breaking my heart",
                "i'll never recover",
                "you're killing me",
                "i will hurt myself",
                "i can not go on without you",
                "you are breaking my heart",
                "i will never recover",
                "you are killing me",
                "i thought you loved me",
                "if you care about me",
                "you promised me",
                "after everything we've been through",
                "i gave up everything"
            ],
            
            'financial_abuse': [
                "you can't spend that",
                "you can not spend that"
                "give me your money",
                "you have to ask me first",
                "i'll handle the money",
                "i will handle the money",
                "you're bad with money",
                "you are bad with money",
                "you can't be trusted",
                "you can not be trusted",
                "you owe me everything",
                "i paid for everything",
                "you're financially dependent",
                "you can't make it without me",
                "you are financially dependent",
                "you can't make it without me"
            ]
        }
        
        
        self.positive_patterns = {
            'support': [
                'proud of you', 'believe in you',
                'can do it', 'support you',
                'here for you', 'help you',
                "i'm listening",
                "you've got this",
                "you're capable",
                "you're strong",
                "i am listening",
                "you have got this",
                "you are capable",
                "you are strong",
                "i believe in you",
                "you can handle this",
                "you are doing great",
                "keep going",
                "i'm here if you need me",
                "i am here if you need me"
            ],
            
            'respect': [
                r'\bunderstand\b', 'your choice',
                'your decision', 'respect your',
                'your space', 'up to you',
                "it's your decision",
                "it is your decision",
                "i respect that",
                "take your time",
                "no pressure",
                "whatever you decide",
                "i trust your judgment",
                "your boundaries matter",
                "you know what's best for you"
            ],
            
            'appreciation': [
                'thank you', r'\bgrateful\b',
                r'\bappreciate\b', 'value you',
                'mean to me', 'care about you',
                "you're important",
                "you are important",
                "i appreciate you",
                "you matter to me",
                "you make a difference",
                "i'm thankful for you",
                "i am thankful for you",
                "you're special to me",
                "you are special to me"
            ],
            
            'empathy': [
                "i understand how you feel",
                "that must be difficult",
                "i hear you",
                "your feelings are valid",
                "that makes sense",
                "i can see why you feel that way",
                "you have every right to feel",
                "that sounds challenging",
                "i'm sorry you're going through this"
            ],
            
            'encouragement': [
                "you're making progress",
                "you are making progress",
                "keep going",
                "you're getting better",
                "you are getting better",
                "small steps count",
                "you're growing",
                "you're learning",
                "you're improving",
                "you've come so far",
                "you are growing",
                "you are learning",
                "you are improving",
                "you have come so far",
                "you should be proud"
            ],
            
            'healthy_boundaries': [
                "take your time",
                "no pressure",
                "when you're ready",
                "it's okay to say no",
                "you don't have to",
                "when you are ready",
                "it is okay to say no",
                "you do not have to",
                "your choice matters",
                "you decide",
                "whatever works for you",
                "what do you think?"
            ],
            
            'constructive_communication': [
                "let's talk about it",
                "i want to understand",
                "help me understand",
                "can we discuss",
                "what are your thoughts",
                "how do you feel about",
                "i'd like to hear your perspective",
                "i would like to hear your perspective",
                "let's find a solution together"
            ]
        }
        
        
        import re
        self.compiled_patterns = {
            'toxic': {
                category: [re.compile(pattern, re.IGNORECASE) 
                          for pattern in patterns]
                for category, patterns in self.toxic_patterns.items()
            },
            'positive': {
                category: [re.compile(pattern, re.IGNORECASE) 
                          for pattern in patterns]
                for category, patterns in self.positive_patterns.items()
            }
        }

    def analyze_message(self, message: str) -> Dict[str, Any]:
        
        message = message.lower()
        
        
        toxic_matches = {}
        for category, patterns in self.compiled_patterns['toxic'].items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(message)
                if found:
                    matches.extend(found)
            if matches:
                toxic_matches[category] = list(set(matches))
        
        
        positive_matches = {}
        for category, patterns in self.compiled_patterns['positive'].items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(message)
                if found:
                    matches.extend(found)
            if matches:
                positive_matches[category] = list(set(matches))
        
        
        has_toxic = bool(toxic_matches)
        has_positive = bool(positive_matches)
        
        if has_toxic:
            prediction = "Red Flag"
            confidence = 75.0
        elif has_positive:
            prediction = "Green Flag"
            confidence = 70.0
        else:
            prediction = "Neutral"
            confidence = 50.0
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'toxic_matches': toxic_matches,
            'positive_matches': positive_matches
        }
