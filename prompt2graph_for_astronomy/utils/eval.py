from utils import call_llm_api

class Eval:
    def __init__(self):
        self.llm_client = call_llm_api.LLMCompletionCall()
        
    def eval(self, question, gold_answer, answer):
        prompt = f"""
        You are an expert evaluator. Your task is to determine if the predicted answer is correct based on the question and gold answer.
        The criteria should be reasonable, not too strict or too lenient.
        
        Question: {question}
        Gold Answer: {gold_answer}
        Predicted Answer: {answer}
        
        Return only "1" (correct) or "0" (incorrect):
        """
        return self.llm_client.call_api(prompt)
