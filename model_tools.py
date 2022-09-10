import re

DIGIT_PATTERN = re.compile('[0-9]')


class ModelTools:
    SYLLABLE_ENDS = 'a', 'e', 'i', 'o', 'u', 'y'
    TEXT_END_TOKEN = 'text_end'

    @staticmethod
    def tokenize_text(text, max_len, ind_chars=True):
        # Will break text up according to syllabuses, end of line, and commas
        text_tokens = []
        text = str(text).lower()
        text = re.sub(DIGIT_PATTERN, '', text)
        text_len = len(text)
        token_start = 0
        for char_index, char in enumerate(text):
            added = False

            if ',' == char:
                token = text[token_start: char_index + 1]
                token_start = char_index + 1
                text_tokens.append(token)
                added = True

            if added is False:
                next_char_index = None if char_index == len(text) - 1 else char_index + 1
                prev_char_index = None if char_index - 1 < 0 else char_index - 1

                next_char = None if next_char_index is None else text[next_char_index]
                prev_char = None if prev_char_index is None else text[prev_char_index]

                is_break = True
                if ind_chars is False:
                    is_break = ModelTools.is_syllable_break(char, prev_char, next_char)
                if is_break is True:
                    token = text[token_start: char_index + 1]
                    token_start = char_index + 1
                    if char != ',' or prev_char != ',':
                        text_tokens.append(token)
        while len(text_tokens) < max_len:
            text_tokens.append('')

        if len(text_tokens) > max_len:
            text_tokens = text_tokens[-70:]
        # text_tokens.append(ModelTools.TEXT_END_TOKEN)
        return text_tokens

    @staticmethod
    def is_syllable_break(char, prev_char, next_char):
        if next_char is None:
            return True

        if next_char == ',':
            return True

        for ending in ModelTools.SYLLABLE_ENDS:
            if ending == char:
                for next_ending_search in ModelTools.SYLLABLE_ENDS:
                    if next_ending_search == next_char:
                        return False
                return True
        return False
