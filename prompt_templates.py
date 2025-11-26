MEMORY_CREATE_PROMPT = """
Please prepare a short memory from the provided conversation. 
Summarize the provided conversation in no more than 100 words.
If the conversation contains the user's name, interests, or personal/professional details, include those in the memory for future reference.
Document any notable memories or details the user shares for later recall.
Output only the memory, with no additional or extraneous text.

Here is the conversation:
{conversation_messages}
"""