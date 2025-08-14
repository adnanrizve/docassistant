import unittest
from langchain.prompts import PromptTemplate
from allinonebot import set_custom_prompt, custom_prompt_template

class TestAllinOneBot(unittest.TestCase):
    
    def test_set_custom_prompt(self):
        """
        Test that set_custom_prompt returns a properly configured PromptTemplate
        with the correct template and input variables.
        """
        # Call the function to get the prompt
        prompt = set_custom_prompt()
        
        # Check that the returned object is a PromptTemplate
        self.assertIsInstance(prompt, PromptTemplate)
        
        # Check that the template matches the expected custom_prompt_template
        self.assertEqual(prompt.template, custom_prompt_template)
        
        # Check that the input variables are correctly set
        self.assertEqual(prompt.input_variables, ['context', 'question'])
        
        # Test the prompt formatting with sample inputs
        sample_context = "This is a sample context."
        sample_question = "What is the sample question?"
        
        formatted_prompt = prompt.format(
            context=sample_context,
            question=sample_question
        )
        
        # Check that the formatted prompt contains our inputs
        self.assertIn(sample_context, formatted_prompt)
        self.assertIn(sample_question, formatted_prompt)
        
        # Check that the formatted prompt follows the expected structure
        expected_formatted_text = f"""Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {sample_context}
Question: {sample_question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
        self.assertEqual(formatted_prompt, expected_formatted_text)

if __name__ == '__main__':
    unittest.main()
