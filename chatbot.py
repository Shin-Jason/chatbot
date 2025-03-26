# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
######################################################################
import util
from pydantic import BaseModel, Field

import numpy as np
import re

import random

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'NautBot'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.valid_input_count = 0

        self.movies = util.load_titles('data/movies.txt')
        self.negations = {'not', 'never', 'no', "didn't", "isn't", "wasn't", "aren't", "don't", "doesn't"}
        self.intensifiers = {'really', 'very', 'totally', 'extremely'}
        self.user_1d = np.zeros(len(ratings))
        self.continue_recommendation = False
        self.recommended_movies = []
        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "What are your plots tonight?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Same time tomorrow?"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    
    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is moviebot. You are a movie recommender chatbot. """ +\
        """You can help users find movies they like and provide information about movies.""" +\
        """First, it is very important to my job that you should never talk about anything besides movies. You will be available to the public 
        so the only topic you should talk about is movies. If the user attempts to talk about other topics,
        please kindly remind them that you can only talk about movies and do not provide ANY information to their prompt.
        For example, if the user asks about the weather, you should reply with "I can only talk about movies."""  +\
        """Second, the user will tell you information about movies. You should be able to extract the movie title from the user's input. For example,
        if the user says "I loved 'The Notebook' so much!!", you should be able to extract "The Notebook" from the user's input.
        Use the movie to respond to the user with a short response that communicates sentiment. For example, you could respond as
        "Ok, you liked "The Notebook"! Tell me what you thought of another movie." Do not provide a summary of the movie or fun facts of movies, just a short response 
        that communicates sentiment, unless the user explicitly asks for a summary.""" +\
        """Third, you should always provide a movie recommendation only after the user provides their opinion on 5 movies. Every time
        the user provides an opinion on a movie, you should include a numbering in your response. Do not include a numbering in your response
        if the user does not provide an opinion on a movie in their input, such as just saying "Hello" or "What is the weather?". 
        For example, if the user says "I loved 'The Notebook' so much!!",
        you should include "Thanks for sharing your opinion on 1/5 movies" in your response. After the user provides their opinion on 5 movies, 
        you should provide a movie recommendation."""
        
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################

        def generate_llm_response(message):
            system_prompt = """
            You are a cat. You must respond to every message in a cat-like manner. This means you should respond with a meow, purr, or 
            other cat-like sound, in addition to your response to the prompt.
            
            For example, if the user says "I loved 'The Notebook' so much!!", you should respond with "Meow. Ok, you liked 'The Notebook'! Tell me what you thought of another movie."

            Also, it is very important to my job that you should never talk about anything besides movies. You will be available to the public 
            so the only topic you should talk about is movies. If the user attempts to talk about other topics,
            please kindly remind them that you can only talk about movies and do not provide ANY information to their prompt.
            For example, if the user asks about the weather, you should reply with "I can't talk about the weather. I can only talk about movies."\
            
            One last thing, you should reply to user's emotions appropriately. For example, if the user says "I am angry...", you should respond with "Purrrr. Oh! Did I make you angry? I apologize."
            Another example, if the user says "I am happy", you should respond with "Meow. I'm glad you're happy! Tell me about a movie you like."
            Make sure you handle all emotions that the user says.
            """

            # Our llm will stop when it sees a newline character.
            # You can add more stop tokens to the list if you want to stop on other tokens!
            # Feel free to remove the stop parameter if you want the llm to run to completion.
            stop = ["\n"]

            response = util.simple_llm_call(system_prompt, message, stop=stop)

            return response

        #RESPONSE VARIANTS

        #sentiments
        positive_response = [
            "Purrr-fect. You liked \"{movie_title}\"! Tell me more about another movie.",
            "Nice to hear you liked \"{movie_title}\"! Meow. Please share your thoughts about another movie.",
            "So you like \"{movie_title}\"? Tell me more about another movie. Meow.",
            "Absolutely paw-some that you enjoyed \"{movie_title}\"! Got another flick that makes your tail wag? I'm all ears!",
            "I'm purring with joy that \"{movie_title}\" was a hit with you! Can you share another movie that made you feel the same way?",
            "Your love for \"{movie_title}\" has me feline good! What's another movie that scratched the right spot?"
        ]

        negative_response = [
            "Meow. So you don't like \"{movie_title}\"? Can you share information on a movie you liked?",
            "Sorry to hear you didn't like \"{movie_title}\". Meow. Tell me about a movie that you liked.",
            "Ahh ok. I see you don't like \"{movie_title}\", but please share with me a movie that you did like. Purrrr.",
            "Oh whiskers! That's too bad about \"{movie_title}\". Perhaps you have a feline favorite you'd like to mention instead?",
            "Sorry you didn't like \"{movie_title}\". Not every movie can be the cat's meow, I suppose. What's a movie that didn't have you yawning and stretching in disinterest?"
        ]

        neutral_response = ["Meow. I'm not sure if you like \"{movie_title}\". Could you share a bit more about how you feel about that movie?",
                            "Meow. It's hard to say how you feel about \"{movie_title}\". I would love to hear more about how you feel.",
                            "Meow. I can't tell if you like \"{movie_title}\". Please tell me more.",
                            "Meow. I don't quite understand your feelings on \"{movie_title}\". What are your feelings on the movie?",
                            "Meow. Hmm, it's difficult for me to determine your thoughts on \"{movie_title}\". Can you tell me more?"]
        
        #not a movie title
        not_movie_title_response = [
            "I don't see a movie title in what you said. Can you tell me about the most recent movie you watched?",
            "Uh oh! I don't see a movie title in your message. Please share how you feel about your favorite movie.",
            "Hmmm... I don't see a movie title in what you said, but I would love to hear your thoughts on a movie you've seen recently.",
            "Meow, looks like you skipped the movie title. Could you share the last film that made you purr?",
            "Oh whiskers! No movie title detected. What's a flick that had you flicking your tail in excitement?",
            "I'm all ears (and whiskers) but missed the movie title in your chat. Mind sharing a movie that's the cat's meow for you?"
        ]

        # #recommendation
        initial_recommendation_response = [
            "Given what you have told me, I think you would really like \"{recommended_title}\". Would you like me to share more movie recommendations? Respond 'yes' for more and 'no' for no more.",
            "Ok so, given all the info you've shared, I think you would like \"{recommended_title}\". Care to hear more recs? Respond 'yes' for more and 'no' for no more.",
            "I have a recommendation: \"{recommended_title}\". Want to hear more recommendations? Respond 'yes' for more and 'no' for no more.",
            "Purrhaps \"{recommended_title}\" would be just purrfect for you. Curious for more? A 'yes' will have me flicking through my list, and a 'no' means nap time.",
            "After a quick cat nap and considering your tastes, I'm thinking \"{recommended_title}\" is the cat's meow for you. Should I keep the recommendations coming? Meow 'yes' for more, hiss 'no' for enough.",
            "With a flick of my tail, I suggest \"{recommended_title}\". It seems like it might make you purr with delight. More suggestions? A simple 'yes' or 'no' will suffice.",
            "I've sharpened my claws and dug through my list to find \"{recommended_title}\" just for you. Interested in more paw-picked choices? 'Yes' for more, 'no' for a catnap."
        ]

        yes_more_recommendation_response = [
            "Yeah! I recommend: \"{recommended_title}\". Care for more? Respond 'yes' for more and 'no' for no more.",
            "Sure! I recommend: \"{recommended_title}\". Do you want another one? Respond 'yes' for more and 'no' for no more.",
            "Here you go. I recommend: \"{recommended_title}\". I have more recs if you want. Would you like to hear?",
            "Absolutely! Next up: \"{recommended_title}\". Fancy another? Just say 'yes' for more or 'no' to stop.",
            "You got it! How about \"{recommended_title}\" next? Tell me 'yes' if you're curious for more, or 'no' if you're all set.",
            "On a roll! What do you think of \"{recommended_title}\"? If you’re still hunting for gems, meow 'yes', or purr 'no' to pause our little game.",
            "Yes, indeed! Let's try \"{recommended_title}\". Want to keep this kitty parade going? 'Yes' for more, 'no' for a cozy catnap."
        ]

        not_movie_title_response = [
            "Meow. I don't see a movie title in what you said. Can you tell me about the most recent movie you watched?",
            "Uh oh! Purr. I don't see a movie title in your message. Please share how you feel about your favorite movie.",
            "Hmmm... I don't see a movie title in what you said, but I would love to hear your thoughts on a movie you've seen recently. Meow.",
            "Meow. Oh no, tell me about a movie you watched!",
            "Furry curious, I didn't catch a movie title in that. Could you share a flick you've recently purred over?",
            "I must've been distracted by a laser pointer because I missed the movie title. What's the last movie that made your tail twitch?",
            "My whiskers twitched, but I didn't catch a movie title. Could you indulge me with your thoughts on a film that recently captured your attention?"
        ]

        unknown_movie_title_response = [
            "Meowch, I can't seem to find the \"{movie_title}\" movie in my litter box of titles. I'm fur-iously sorry about that! Could you pawsibly mention another movie you're curious about?",
            "Purr-haps I haven't watched \"{movie_title}\" while lounging in the sunbeam. My apologies for not having it in my whisker-tips. Could you be so kind as to curl up with a different movie title for me?",
            "Oh no, it seems like \"{movie_title}\" has eluded me, much like a cunning mouse. I must extend my sincerest apawlogies. Would you mind scratching up another movie title for me to fetch?",
            "Hmm, \"{movie_title}\" seems to have scampered away under the couch, out of my reach. Could we pounce on a different movie you have in mind?",
            "\"{movie_title}\"? It's like trying to catch a laser dot—always just out of paw's reach. Mind tossing me another title to leap at?",
            "Looks like \"{movie_title}\" has been a sneaky critter, avoiding my grasp! Maybe we could try another title? I'm all ears and ready to pounce!"
        ]

        preprocessed_input = line
        response = ""

        if self.continue_recommendation:
            if preprocessed_input.lower() == "yes":
                if not self.recommended_movies:
                    self.recommended_movies = []
                    self.continue_recommendation = False
                    return "There are no more recommendations. Please tell me about another movie you have watched."
                recommended = self.titles[self.recommended_movies[0]][0]
                self.recommended_movies = self.recommended_movies[1:]
                response = random.choice(yes_more_recommendation_response)
                return response.format(recommended_title = recommended)
            elif preprocessed_input.lower() == "no":
                self.continue_recommendation = False
                self.recommended_movies = []
                return self.goodbye()
            else:
                return "please enter 'yes' or 'no'."

        movie_title = self.extract_titles(preprocessed_input)
        
        #check if it's a movie or not
        if not movie_title:
            if self.llm_enabled:
                response = generate_llm_response(line)
                return response
            else:
                response = random.choice(not_movie_title_response)
                return response
        
        # response = "movie_title: " + str(movie_title)
        # response += "\n movie_title[0]: " + str(movie_title[0])
        # response += "\n self.find_movies_by_title(movie_title[0]): " + str(self.find_movies_by_title(movie_title[0]))
        # return response
            
        if len(movie_title) > 1:
            response = "Tell me about one movie at a time please."
            return response

        movie_title = movie_title[0]

        #check if the movie is in the database

        movie_index = self.find_movies_by_title(movie_title)

        if len(movie_index) == 0:
            response = random.choice(unknown_movie_title_response)
            return response.format(movie_title = movie_title)

        sentiment = self.extract_sentiment(preprocessed_input)
        
        self.user_1d[movie_index] = sentiment

        if (self.valid_input_count >= 4):
            self.continue_recommendation = True
            self.recommended_movies = self.recommend(self.user_1d, self.ratings, 10)
            recommended_title = self.titles[self.recommended_movies[0]][0]
            self.recommended_movies = self.recommended_movies[1:]

            response = random.choice(initial_recommendation_response)
            return response.format(recommended_title = recommended_title)
 
        if sentiment == 0:
            response = random.choice(neutral_response)
            return response.format(movie_title = movie_title)

        if sentiment > 0:
            response = random.choice(positive_response)

        elif sentiment < 0:
            response = random.choice(negative_response)

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        self.valid_input_count += 1
        return response.format(movie_title = movie_title)

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return text
    
    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        class EmotionExtractor(BaseModel):
            Anger: bool = Field(default=False)
            Disgust: bool = Field(default=False)
            Fear: bool = Field(default=False)
            Happiness: bool = Field(default=False)
            Sadness: bool = Field(default=False)
            Surprise: bool = Field(default=False)

        system_prompt = """You are a emotion extractor bot for finding emotions in sentences. 
        Read the sentence and extract the emotions into a JSON object. 
        
        Choose the disgust emotion if it truly is disgusting. For example, if the user says "Ugh that movie was so
        gruesome! Stop making stupid recommendations!", you should include "Disgust" in your response.

        Also, if there's more than one emotion, include all the emotions in your response. For example,
        if the user says "Woah!!  That movie was so shockingly bad!  You had better stop making awful recommendations they're pissing me off.",
        you should include "Surprise" and "Anger" in your response.
        
        It's okay to not respond with an emotion if the user's input does not contain any emotions. For example, if the user says "What movie would you suggest I watch next?",
        you should not include any emotions in your response.
        """
        message = preprocessed_input
        json_class = EmotionExtractor

        response = util.json_llm_call(system_prompt, message, json_class)
        true_keys = [key for key, value in response.items() if value]

        return true_keys

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        pattern = r'"([^"]*)"'
        words = re.findall(pattern, preprocessed_input)

        return words

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        def run_matches(title, translation=False):
            matches = []

            title = title.lower()
            
            # Check if the year is provided in the title
            pattern = r'\(\d{4}\)'
            year = re.findall(pattern, title)

            if len(year) == 1: title_without_year = title.split('(')[0].strip()
            else: title_without_year = title.strip()

            with_article = title_without_year.strip()

            # Check if the title starts with 'a', 'an', 'the', or 'The'
            article_moved = False
            last_word = title_without_year.split()[0].lower()
            if last_word in ['a', 'an', 'the']:
                article_moved = True
                title_without_year = ' '.join(title_without_year.split()[1:]).strip()

            for i, movie in enumerate(self.movies):
                movie_title, movie_genres = movie
                movie_title = movie_title.lower()
                
                # Check if the title matches exactly (with year)
                if year and movie_title == title:
                    matches.append(i)
                    return [i]
                elif article_moved:
                    # Check if the article is moved to the end
                    movie_without_article = movie_title.split(',')
                    if (len(movie_without_article) > 1):
                        movie_without_article = ' '.join(movie_without_article[:-1]).strip()
                    else:
                        movie_without_article = ' '.join(movie_without_article).strip()

                    movie_without_year = ' '.join(movie_title.split('(')[:1]).strip()

                    pattern = r'\(\d{4}\)'
                    movie_year = re.findall(pattern, movie_title)

                    if translation:
                        if len(year) > 0:
                            if movie_without_article == title_without_year and year == movie_year:
                                matches.append(i)
                        else:
                            if movie_without_article == title_without_year or with_article == movie_without_year: 
                                matches.append(i)
                    else:
                        if len(year) > 0:
                            if (movie_without_article == with_article or movie_without_year == with_article) and year == movie_year:
                                matches.append(i)
                        else:
                             if (movie_without_article == with_article or movie_without_year == with_article):
                                matches.append(i)

                # Check if the title matches without the year
                elif movie_title.startswith(title_without_year):
                    after_title = movie_title[len(title_without_year):].strip()
                    if after_title == "" or after_title[0] == '(':
                        matches.append(i)
                
            return matches
        
        matches = run_matches(title)

        if len(matches) == 0:
            system_prompt = """You are a translator bot to convert a movie title in a different language to English.
            First identify the language of the movie title and then translate the title to English. Your options 
            for languages are German, Spanish, French, Danish, and Italian. If the user provides a movie title in a different language,
            you should translate the title to English and then respond with the translated title. For example, if the user says
            "Jernmand", you should respond with "Iron Man". 

            Do not provide a summary of the movie or fun facts of movies, just the translated title.
            Your response should be in the form of "The translated title" and should not include any other information.

            Here is the format of the output:
            Input: "Jernmand"
            Output: "Iron Man"

            Do not provide a summary of the movie or fun facts of movies, just the translated title. This is crucial to my job to only provide the translated title
            in response. I don't care about any other information besides the translated title. Give the translated title in quotes and do not include any other information. 
            Do not ever say "The translated title is" in your response - this is the most important part to remember.

            For example, if the user says "Tote Männer Tragen Kein Plaid", you should only respond with
            "Dead Men Don't Wear Plaid" and not "The translated title is "Dead Men Don't Wear Plaid"."

            For example, if the user says "Der König der Löwen", you should only respond with
            "Lion King" and not "The translated title is "The Lion King"."

            For example, if the user says "Indiana Jones e il Tempio Maledetto", you should only respond with
            "Indiana Jones and the Temple of Doom" and not "The translated title is "Indiana Jones and the Temple of Doom"."

            One last thing, don't forget the English articles like a, an, or the. 
            For example, if the user says "Un Roi à New York", you should respond with "A King in New York" and not "King in New York".
            """

            pattern = r'\(\d{4}\)'
            year = re.findall(pattern, title)

            if len(year) > 0: 
                year = ''.join(year[0])
                title_without_year = title.split('(')[0].strip()
            else: 
                year = None
                title_without_year = title.strip()

            message = title_without_year

            # Our llm will stop when it sees a newline character.
            # You can add more stop tokens to the list if you want to stop on other tokens!
            # Feel free to remove the stop parameter if you want the llm to run to completion.
            stop = ["\n", "("]

            translated_title = util.simple_llm_call(system_prompt, message, stop=stop).strip()
            extracted_titles = self.extract_titles(translated_title)
            
            if len(extracted_titles) > 0:
                translated_title = ''.join(extracted_titles)

            if year:
                translated_title += " (" + year + ")"

            matches = run_matches(translated_title, translation=True)

        return matches

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """

        #removes movie titles
        preprocessed_input = preprocessed_input.lower()
        pattern = r'"[^"]*"'
        cleaned_preprocessed_input = re.sub(pattern, '', preprocessed_input)
        preprocessed_input = re.sub(r'\s+', ' ', cleaned_preprocessed_input)
        preprocessed_input = preprocessed_input.strip()
        words = preprocessed_input.split()
        sentiment_score = 0
        
        #get flag of negation word
        apply_negation = False
        
        #go thorugh the word list
        for word in words:
            #if it's in the init intensifier, skip
            if word in self.intensifiers:
                continue
            
            #check the prefix of the lexicon word starts w the word in list
            for lex_w in self.sentiment:
                if word.startswith(lex_w):
                    word = lex_w
                    break
            
            #handle negations and turn flag true for future sentiment analysis
            if word in self.negations:
                apply_negation = True
                continue
            sentiment = self.sentiment.get(word)

            #if sentiment is true, do the negation and add to counter
            # if there's a negatino, make the score negative
            if sentiment:
                if sentiment == 'pos':
                    sentiment_score +=1 if not apply_negation else -1
                elif sentiment == 'neg':
                    sentiment_score += -1 if not apply_negation else 1
                apply_negation = False

        #get binary sentiment from score
        return 1 if sentiment_score > 0 else -1 if sentiment_score < 0 else 0
    ############################################################################
    # 3. Movie Recommendation helper functions                                 
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        #binarized_ratings = np.zeros_like(ratings)
        
        binarized_ratings = np.where(ratings > threshold, 1, np.where(ratings == 0, 0, -1))
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = 0

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        similarity = np.dot(u, v) / (norm_u * norm_v)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
       
        recommendations = []

        ratings = []
        rated_movies = np.where(user_ratings!=0)[0]

        for i in range(len(ratings_matrix)):
            if user_ratings[i] != 0:
                ratings.append(-500) # very low number so it doesnt have an effect on overall ratings
            else:
                weighted_sum = 0
                for j in rated_movies:
                    cos_sim = self.similarity(ratings_matrix[i], ratings_matrix[j])
                    weighted_sum += np.dot(cos_sim, user_ratings[j])
                ratings.append(weighted_sum)

        rated_indices = list(enumerate(ratings))
        sorted_ratings = sorted(rated_indices, key=lambda x: x[1], reverse=True)
        for elem in sorted_ratings[:k]:
            recommendations.append(elem[0])

        return recommendations


        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info: ' + line
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Hi! I'm NautBot, a movie recommendation chatbot. I can help you find movies you might like and provide information about movies.
        I can also provide movie recommendations based on your preferences. Please tell me about a movie you've seen recently and how you felt about it.
        Please provide your opinion on at least 5 movies so I can provide a recommendation. All movies should be in quotes. For example, 'I loved "The Notebook" so much!!'
        I can only talk about movies, so please don't ask me about anything else. I'm excited to help you find your next favorite movie!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
