BASE_PROMPT =  ("Classify the following political statement into one of these labels:\n"
                "0 = false\n"
                "1 = half-true\n"
                "2 = mostly-true\n"
                "3 = true\n"
                "4 = barely-true\n"
                "5 = pants-fire\n"
                "Respond with only the number (0–5) corresponding to the label. Do NOT respond with anything other than the number.\n"
                "You are limited to producing one token (the label number). If you produce anything else, you will fail.\n"
                "Statement: {statement}\n"
                "Label number:")

ASSISTANT_PROMPT = "tbd"