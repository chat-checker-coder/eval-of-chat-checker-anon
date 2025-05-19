# Evaluation Code for the ChatChecker framework
ChatChecker framework: https://github.com/chat-checker-coder/chat-checker-anon

## Setup
1. Setup virtual environment: `python -m venv venv`
2. Activate virtual environment: `source venv/bin/activate` (Windows: `venv\Scripts\activate`)
3. Install requirements: `pip install -r requirements.txt`
4. Install pre-commit hooks: `pre-commit install`

## ChatChecker Framework Evaluation
### Evaluation of the Dialogue Breakdown Detector
#### Breakdown Detection
Follow the instructions in `benchmarks/benchmark_breakdown_detection.ipynb` and `benchmarks/investigate_breakdown_detector_results.ipynb` to evaluate the dialogue breakdown detector.

#### Error Type Classification
Follow the instructions in `benchmarks/benchmark_error_classification.ipynb` to evaluate the error type classifier.

### Evaluation of the Dialogue Rater
Follow the instructions in `benchmarks/benchmark_dialogue_rating.ipynb` to evaluate the dialogue rater.

### Evaluation of the User Simulators

The results reported in the thesis can be found in the `user_simulation_results` directory. The were collected using the `collect_simulation_stats.py` script based on the runs with the respective chatbots in  `chabots` directory.

Follow the instructions below to run `chat_checker` against the target dialogue systems and reproduce the results.

#### General Chat-Checker Setup
1. Make sure the target chatbots are registered in `chat_checker` by running `chat-checker register -d chatbots`.
2. Make sure that you have configured the environment variables as described in the `README.md` of the ChatChecker repository.
 
#### Against the AutoTOD Dialogue System
0. Clone our AutoTOD fork: https://github.com/chat-checker-coder/autotod-fork-anon
1. Specify the desired base LLM for AutoTOD in the `chatbots/autotod_multiwoz/chatbot_client.py` file.
2. Follow the instructions in the AutoTOD README to install the dependencies and run the chatbot server.
3. Optional: if you want to evaluate the AutoTOD user simulator also follow the instructions in the AutoTOD README to run the user simulator server.
4. Reuse the user personas used in our experiments by copying the contents of `user_personas_paper_run_X` to a fresh `user_personas` directory in `chatbots/autotod_multiwoz`. Alternatively you can generate your own user personas using `chat-checker generate-personas autotod_multiwoz -t <persona_type> -n <number_of_personas>`.
5. Start a run manually (`chat-checker run autotod_multiwoz -u <user_type>`) or follow the commands we used for our experiments (see commands in `paper_run_commands.txt`).
6. Inspect the results in the directory of the respective run. You can aggregate the results using the `collect_simulation_stats.py` script.

#### Against the Goal-Setting Assistant
0. Clone the version of our Goal-Setting Assistant that we used for our experiments (<github-url-redacted-for-anonymity>)
1. Follow the instructions in the Goal-Setting Assistant README to install the dependencies and run the chatbot server. IMPORTANT: Make sure to run it on port 5000.
2. Reuse the user personas used in our experiments by copying the contents of `user_personas_paper_run_X` to a fresh `user_personas` directory in `chatbots/study_goal_assistant`. Alternatively you can generate your own user personas using `chat-checker generate-personas study_goal_assistant -t <persona_type> -n <number_of_personas>`.
3. Start a run manually (`chat-checker run study_goal_assistant -u <user_type>`) or follow the commands we used for our experiments (see commands in `paper_run_commands.txt`).
4. Inspect the results in the directory of the respective run. You can aggregate the results using the `collect_simulation_stats.py` script.
