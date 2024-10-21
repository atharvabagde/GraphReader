
from openai import OpenAI



class OpenAI_client:
    """
    A custom client class for interacting with the OpenAI API.

    Attributes:
    ----------
    api_key : str
        The API key for accessing OpenAI services.
    model : str
        The model name used for generating responses (e.g., "gpt-3.5-turbo").
    temperature : float
        The temperature setting controls the randomness of the model's output.
    sys_prompt : str or bool
        The system prompt providing initial instructions to the model.
    user_prompt : str
        The user query or input passed to the model.
    message : list
        A list of message dictionaries that define the conversation context.

    Methods:
    -------
    __init__(**kwargs)
        Initializes the OpenAI client with API key and optional parameters.
        
    __repr__()
        Returns a string representation of the OpenAI_client object.
        
    _construct_message()
        Constructs the message payload based on system and user prompts.
        
    get_response(query, **kwargs)
        Generates a response from the model based on the provided query and settings.
    """
    def __init__(self, **kwargs):
        # Initialize with API key, either from kwargs or environment variable
        self.api_key = kwargs.get('api_key')
        if not self.api_key:
            raise ValueError("API key is missing. Provide it as 'api_key' or set the 'OPENAI_API_KEY' environment variable.")

    def __repr__(self):
        return "OpenAI API custom client"

    def _construct_message(self):
        # Constructs the message based on system and user prompts
        self.sys_dict = {"role": "system", "content": self.sys_prompt} if self.sys_prompt else None
        self.user_dict = {"role": "user", "content": self.user_prompt}
        
        # Build the final message list
        self.message = [self.sys_dict, self.user_dict] if self.sys_dict else [self.user_dict]

    def get_response(self, query, **kwargs):
        # Set model, temperature, and prompts from kwargs or use defaults
        self.model = kwargs.get('model', "gpt-3.5-turbo")
        self.temperature = kwargs.get('temperature', 0.7)
        self.sys_prompt = kwargs.get('sys_prompt', None)
        self.user_prompt = query
        
        # Construct the message for the OpenAI API call
        self._construct_message()
        
        # Call OpenAI's API to get the response
        try:
            client = OpenAI(api_key = self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=self.message,
                temperature=self.temperature
            )
            return response.choices[0].message.content  # Extract and return the response text
        except Exception as e:
            print(f"Error during API call: {e}")
            return None