import Levenshtein

def calculate_cer(pred: str, target: str) -> float:
    """
    Calculates the Character Error Rate (CER).
    CER = (Substitutions + Insertions + Deletions) / Number of Characters in Target
    """
    if len(target) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    
    distance = Levenshtein.distance(pred, target)
    cer = distance / len(target)
    return cer

def calculate_wer(pred: str, target: str) -> float:
    """
    Calculates the Word Error Rate (WER).
    """
    pred_words = pred.split()
    target_words = target.split()
    
    if len(target_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
        
    # We use character distance on words array by mapping them to chars, 
    # but for simplicity in this implementation we just map words to characters temporarily.
    # A proper WER uses dynamic programming on the word list.
    
    # Simple dynamic programming for WER
    d = [[0] * (len(target_words) + 1) for _ in range(len(pred_words) + 1)]
    
    for i in range(len(pred_words) + 1):
        d[i][0] = i
    for j in range(len(target_words) + 1):
        d[0][j] = j
        
    for i in range(1, len(pred_words) + 1):
        for j in range(1, len(target_words) + 1):
            if pred_words[i-1] == target_words[j-1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i-1][j] + 1,       # Insertion
                d[i][j-1] + 1,       # Deletion
                d[i-1][j-1] + cost   # Substitution
            )
            
    wer = d[len(pred_words)][len(target_words)] / len(target_words)
    return wer
