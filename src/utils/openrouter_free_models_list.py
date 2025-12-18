# # # get_free_models.py
# #
# # import requests
# # import json
# #
# #
# # def get_openrouter_free_models_from_api():
# #     """
# #     Fetches the list of models directly from the OpenRouter API and filters for
# #     the ones that are free to use. This is much more reliable than scraping.
# #     """
# #     api_url = "https://openrouter.ai/api/v1/models"
# #
# #     print(f"Fetching model data from API: {api_url}")
# #
# #     try:
# #         response = requests.get(api_url, timeout=15)
# #         response.raise_for_status()  # Raise an exception for bad status codes
# #
# #         # The response is a JSON object, so we parse it directly
# #         data = response.json()
# #
# #         # The list of models is inside the 'data' key
# #         models_list = data.get('data', [])
# #
# #         if not models_list:
# #             print("Error: Could not find the 'data' key in the API response.")
# #             return None
# #
# #         # Filter for models where the completion cost is '0'
# #         # Note: The cost is a string in the JSON, so we check against '0'
# #         free_model_ids = [
# #             model['id'] for model in models_list
# #              if model.get('pricing', {}).get('completion') == '0' and
# #                 model.get('pricing', {}).get('prompt') == '0' and
# #                "structured_outputs" in model.get('supported_parameters', {})
# #
# #         ]
# #
# #         return sorted(free_model_ids)
# #
# #     except requests.exceptions.RequestException as e:
# #         print(f"Error: Could not fetch data from the API. {e}")
# #         return None
# #     except (json.JSONDecodeError, KeyError) as e:
# #         print(f"Error: Could not parse the JSON response or find the required keys. {e}")
# #         return None
# #
# #
# # if __name__ == "__main__":
# #     model_ids = get_openrouter_free_models_from_api()
# #
# #     if model_ids:
# #         print("\n--- Fetched Free Model IDs from API ---")
# #         print(f"Found {len(model_ids)} free models.")
# #
# #         print("\nCopy and paste this list into your benchmark script:")
# #         print("-" * 50)
# #
# #         # Print in a format that's easy to copy into a Python list
# #         print("MODELS_TO_BENCHMARK = [")
# #         for model_id in model_ids:
# #             print(f'    "{model_id}",')
# #         print("]")
# #         print("-" * 50)
#
# import json
#
# import requests
#
# response = requests.get(
#     url="https://openrouter.ai/api/v1/key",
#     headers={
#         "Authorization": f"Bearer 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"
#     }
# )
#
# print(json.dumps(response.json(), indent=2))
