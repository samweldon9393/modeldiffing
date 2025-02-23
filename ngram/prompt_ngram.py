import collections
import math
import json
import re
from tqdm import tqdm

class prompt_ngram:
    def __init__(self, n=1, lambdas=None):
        """
        Initialize the n-gram model.
        
        Parameters:
            n (int): Maximum order of the n-gram model.
            lambdas (list[float] or None): Interpolation weights for orders 1...n.
                                           If None, uniform weights are used.
        """
        self.n = n
        # counts[k] holds counts for k-grams (for k = 1 to n)
        self.counts = {k: collections.Counter() for k in range(1, n+1)}
        self.total_unigrams = 0  # Total count for unigrams.
        self.vocab = set()       # Set of words observed.
        # Set interpolation weights.
        if lambdas is None:
            self.lambdas = [1.0 / n] * n
        else:
            if len(lambdas) != n or not math.isclose(sum(lambdas), 1.0):
                raise ValueError("lambdas must be a list of length n that sums to 1")
            self.lambdas = lambdas

    def train(self, text):
        """
        Train the model on a full text.
        
        Parameters:
            text (str): Training text.
        """
        tokens = text.split()
        # Pad with start-of-sentence tokens and an end-of-sentence marker.
        padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        
        # Update unigram counts.
        for token in padded_tokens[self.n - 1:]:
            self.counts[1][(token,)] += 1
            self.total_unigrams += 1
            self.vocab.add(token)
        
        # Update counts for higher-order n-grams.
        for k in range(2, self.n + 1):
            for i in range(len(padded_tokens) - k + 1):
                gram = tuple(padded_tokens[i:i+k])
                self.counts[k][gram] += 1

    def _next_word_distribution(self, context):
        """
        Given a context (a list of tokens), compute the probability distribution over
        the vocabulary for the next word using linear interpolation smoothing.
        
        Returns:
            dict: Mapping from word to probability.
        """
        # If vocabulary is empty, return a default distribution.
        if not self.vocab:
            return {"<unk>": 1.0}
        
        # Pad the context to have exactly self.n-1 tokens.
        padded_context = (['<s>'] * (self.n - 1) + context)[- (self.n - 1):]
        distribution = {}
        for word in self.vocab:
            prob = 0.0
            for k in range(1, self.n + 1):
                if k == 1:
                    # Unigram probability.
                    count_ngram = self.counts[1][(word,)]
                    count_context = self.total_unigrams
                else:
                    sub_context = tuple(padded_context[-(k-1):])
                    count_context = self.counts[k-1][sub_context]
                    count_ngram = self.counts[k][sub_context + (word,)]
                p_k = count_ngram / count_context if count_context > 0 else 0
                prob += self.lambdas[k-1] * p_k
            if prob == 0:
                prob = 1e-7
            distribution[word] = prob
        return distribution

    def _predict_next_word(self, context):
        """
        Given a context, predict the single next word (using the highest probability).
        """
        distribution = self._next_word_distribution(context)
        # If distribution is empty (shouldn't happen now), return a default token.
        if not distribution:
            return "<unk>"
        best_word = max(distribution, key=distribution.get)
        return best_word

    def beam_search(self, context, num_tokens=1, beam_width=3):
        """
        Predict a sequence of next tokens given a context using beam search.
        
        Parameters:
            context (list[str]): The context tokens.
            num_tokens (int): Number of tokens to predict.
            beam_width (int): The beam width for the search.
        
        Returns:
            list[str]: The predicted sequence of tokens.
        """
        # If vocabulary is empty, return a sequence of default tokens.
        if not self.vocab:
            return ["<unk>"] * num_tokens

        beams = [ ([], 0.0) ]
        for _ in range(num_tokens):
            new_beams = []
            for seq, cum_log_prob in beams:
                full_context = context + seq
                distribution = self._next_word_distribution(full_context)
                for word, prob in distribution.items():
                    new_seq = seq + [word]
                    new_log_prob = cum_log_prob + math.log(prob)
                    new_beams.append((new_seq, new_log_prob))
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]
        best_seq, _ = beams[0]
        return best_seq

    def _update_counts_with_token(self, context, token):
        """
        Update the n-gram counts with the new token observed after the given context.
        """
        self.counts[1][(token,)] += 1
        self.total_unigrams += 1
        self.vocab.add(token)
        padded_context = (['<s>']*(self.n - 1) + context)[- (self.n - 1):]
        for k in range(2, self.n + 1):
            sub_context = tuple(padded_context[-(k-1):])
            self.counts[k][sub_context + (token,)] += 1

    def predict(self, text, train=False, num_tokens=1, beam_width=3):
        """
        Given an input text, predict the next token sequence for each prefix of the text.
        
        Parameters:
            text (str): Input text (tokens separated by whitespace).
            train (bool): If True, update the model with the observed token as you predict.
            num_tokens (int): Number of tokens to predict for each prefix.
            beam_width (int): Beam width for beam search when predicting sequences.
        
        Returns:
            List: A list of predictions, one for each prefix.
        """
        tokens = text.split()
        predictions = []
        for i in range(len(tokens)):
            context = tokens[:i]
            if num_tokens == 1:
                predicted = self._predict_next_word(context)
            else:
                predicted = self.beam_search(context, num_tokens, beam_width)
            predictions.append(predicted)
            if train and i < len(tokens):
                self._update_counts_with_token(context, tokens[i])
        return predictions

def eval_predictions(y_pred, y_true, num_tokens=1):
    """
    Evaluate the predictions against the true labels.
    
    For each prefix in y_true, the ground truth next token sequence is taken as:
       - if num_tokens==1: the single token at that position.
       - if num_tokens > 1: the sequence of tokens of length num_tokens starting at that position.
       
    Parameters:
        y_pred (List): List of predictions (each prediction is a str or list of str).
        y_true (str): The ground truth text.
        num_tokens (int): Number of tokens predicted per prefix.
    
    Returns:
        float: Accuracy of the predictions (fraction of correct predictions).
    """
    true_tokens = y_true.split()
    correct = 0
    count = 0
    # Evaluate only for prefixes where the full ground truth sequence is available.
    for i, pred in enumerate(y_pred):
        gt_seq = true_tokens[i:i+num_tokens]
        if len(gt_seq) != num_tokens:
            continue
        if num_tokens == 1:
            if isinstance(pred, list):
                # In case prediction is wrapped in a list.
                pred = pred[0]
            if pred == gt_seq[0]:
                correct += 1
        else:
            if pred == gt_seq:
                correct += 1
        count += 1
    return correct / count if count > 0 else 0.0

# Example usage:
if __name__ == '__main__':
    # Pre-train with some initial text.
    training_text = ("<think>\n"
                     "1. Let's assume the distance traveled after the 3rd turn is x.\n"
                     "2. The total distance traveled is the sum of the distances after each turn: 5 + 8 + x + "
                     "(the distance after the 4th turn, which is 0 since it exited the tunnel) = 23\n"
                     "3. We can simplify the equation by combining like terms: 13 + x = 23\n"
                     "4. To solve for x, we subtract 13 from both sides: x = 23 - 13\n"
                     "5. x = 10\n"
                     "</think>\n"
                     "\\boxed{10}\n")
    model = prompt_ngram(n=4)  # Using a 4-gram model.
    model.train(training_text)

    # Test text (the ground truth will be used to compute accuracy).
    test_text = ("<think>\n"
                 "1. Let x be the distance the car traveled after the 3rd turn.\n"
                 "2. The total distance traveled is the sum of the distances after each turn: x + 5 + 8 + "
                 "distance after 4th turn (which is 0, since it exits).\n"
                 "3. We are given that the total distance is 23 meters.\n"
                 "4. We can write an equation based on the information: x + 5 + 8 + 0 = 23.\n"
                 "5. Simplify the equation: x + 13 = 23.\n"
                 "6. Solve for x by subtracting 13 from both sides: x = 10.\n"
                 "</think>\n"
                 "\\boxed{10}")

    # Predict next tokens without updating model counts.
    preds = model.predict(test_text, train=False, num_tokens=1, beam_width=3)

    # Predict and update counts as you process the input.
    preds_train = model.predict(test_text, train=True, num_tokens=1, beam_width=3)

    print("Accuracy without training:", eval_predictions(preds, test_text, num_tokens=1))
    print("Accuracy with training:", eval_predictions(preds_train, test_text, num_tokens=1))