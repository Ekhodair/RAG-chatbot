from transformers import TextIteratorStreamer


class ResponseTokensStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, input_ids, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.input_length = len(input_ids[0])
    
    def put(self, value):
        if self.next_tokens_are_prompt:
            # Skip tokens that are part of the prompt
            if len(value.tolist()[0]) <= self.input_length:
                self.next_tokens_are_prompt = False
                return
            # Only take the new tokens
            value = value[:, self.input_length:]
            self.next_tokens_are_prompt = False
        super().put(value)