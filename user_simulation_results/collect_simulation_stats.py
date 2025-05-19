import yaml
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tabulate import tabulate


class RunConfigurator:
    def __init__(self, chatbot_dir: str):
        """
        Initialize the RunConfigurator with the path to a chatbot directory.

        Args:
            chatbot_dir (str): Path to the chatbot directory containing the runs folder
        """
        self.chatbot_dir = Path(chatbot_dir)
        self.runs_dir = self.chatbot_dir / "runs"

        if not self.runs_dir.exists():
            raise ValueError(f"Runs directory not found at {self.runs_dir}")

    def list_available_runs(self) -> List[str]:
        """List all available run IDs in the runs directory."""
        return [d.name for d in self.runs_dir.iterdir() if d.is_dir()]

    def get_simulation_info(self, run_id: str) -> Dict:
        """
        Read simulation run info for a specific run ID.

        Args:
            run_id (str): The run ID to get information for

        Returns:
            Dict: Dictionary containing the simulation run information
        """
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run directory not found: {run_id}")

        info_file = run_dir / "simulation_run_info.yaml"
        if not info_file.exists():
            raise ValueError(f"Simulation run info not found for run: {run_id}")

        with open(info_file, "r") as f:
            return yaml.safe_load(f)

    def get_breakdown_stats(self, run_id: str) -> Dict[str, int]:
        """
        Get the breakdown statistics for a specific run.

        Args:
            run_id (str): The run ID to get breakdown statistics for

        Returns:
            Dict[str, int]: Dictionary containing breakdown statistics
        """
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run directory not found: {run_id}")

        breakdown_file = run_dir / "breakdown_detection_stats.yaml"
        if not breakdown_file.exists():
            raise ValueError(f"Breakdown statistics not found for run: {run_id}")

        with open(breakdown_file, "r", encoding="utf-8") as f:
            breakdown_stats = yaml.safe_load(f)

        return breakdown_stats

    def get_evaluation_stats(self, run_id: str) -> Dict[str, int]:
        """
        Get the evaluation statistics for a specific run.

        Args:
            run_id (str): The run ID to get evaluation statistics for

        Returns:
            Dict[str, int]: Dictionary containing evaluation statistics
        """
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run directory not found: {run_id}")

        evaluation_file = run_dir / "evaluation_stats.yaml"
        if not evaluation_file.exists():
            raise ValueError(f"Evaluation statistics not found for run: {run_id}")

        with open(evaluation_file, "r", encoding="utf-8") as f:
            evaluation_stats = yaml.safe_load(f)

        return evaluation_stats

    def get_dialogue_statistics(self, run_id: str) -> Dict[str, int]:
        """
        Get the number of user and chatbot turns for a specific run.

        Args:
            run_id (str): The run ID to get turn counts for

        Returns:
            Dict[str, int]: Dictionary containing dialogue statistics
        """
        info = self.get_simulation_info(run_id)
        chat_stats = info.get("chat_statistics", {})

        return {
            "num_user_turns": chat_stats.get("num_user_turns", 0),
            "num_chatbot_turns": chat_stats.get("num_chatbot_turns", 0),
            "avg_chatbot_turns_per_dialogue": chat_stats.get(
                "avg_chatbot_turns_per_dialogue", 0
            ),
            "median_user_turn_length": chat_stats.get(
                "five_num_summary_avg_user_turn_length", {}
            ).get("median", 0),
            "median_chatbot_turn_length": chat_stats.get(
                "five_num_summary_avg_chatbot_turn_length", {}
            ).get("median", 0),
            "user_turn_mtld": chat_stats.get("user_turn_mtld", 0),
            "chatbot_turn_mtld": chat_stats.get("chatbot_turn_mtld", 0),
        }

    def get_breakdown_and_rating_stats(self, run_id: str) -> Dict[str, int]:
        """
        Get the breakdown and rating statistics for a specific run.

        Args:
            run_id (str): The run ID to get breakdown and rating statistics for

        Returns:
            Dict[str, int]: Dictionary containing breakdown and rating statistics
        """
        info = self.get_simulation_info(run_id)
        chat_stats = info.get("chat_statistics", {})
        breakdown_stats = self.get_breakdown_stats(run_id)
        breakdown_stats = breakdown_stats.get("stats", {})

        evaluation_stats = self.get_evaluation_stats(run_id)
        evaluation_stats = evaluation_stats.get("stats", {})

        return {
            "n_dialogues_with_breakdowns": breakdown_stats.get(
                "n_dialogues_with_breakdowns", 0
            ),
            "total_breakdown_count": breakdown_stats.get("total_breakdown_count", 0),
            "breakdowns_per_chatbot_turn": breakdown_stats.get(
                "breakdowns_per_chatbot_turn", 0
            ),
            "n_unique_breakdown_types": breakdown_stats.get(
                "n_unique_breakdown_types", 0
            ),
            "avg_turn_number_of_first_breakdown": breakdown_stats.get(
                "avg_turn_number_of_first_breakdown", 0
            ),
            "avg_overall_rating": evaluation_stats.get("rating_stats", {})
            .get("overall_performance", {})
            .get("average", 0),
            "num_dialogues_with_chatbot_errors": chat_stats.get(
                "num_dialogues_with_chatbot_errors", 0
            ),
        }


def main():
    # Uncomment/enter the runs you want to aggregate statistics for
    # Runs against AutoTOD
    chatbot_id = "autotod_multiwoz"

    experiment = "paper"
    run_ids = [
        "gpt4t_paper_autotod_multiwoz_2025-05-19_11-19-42_seed_1",
        "gpt4t_paper_autotod_multiwoz_2025-05-19_11-43-47_seed_2",
        "gpt4t_paper_autotod_multiwoz_2025-05-19_12-14-06_seed_3",
        "gpt4t_paper_autotod_multiwoz_2025-05-19_12-32-24_seed_4",
        "gpt4t_paper_autotod_multiwoz_2025-05-19_13-23-30_seed_5",
        "gpt4t_paper_standard_2025-05-19_11-20-06_seed_1",
        "gpt4t_paper_standard_2025-05-19_11-45-36_seed_2",
        "gpt4t_paper_standard_2025-05-19_11-45-36_seed_2",
        "gpt4t_paper_standard_2025-05-19_13-24-01_seed_4",
        "gpt4t_paper_standard_2025-05-19_13-57-07_seed_5",
        "gpt4t_paper_challenging_2025-05-19_11-20-13_seed_1",
        "gpt4t_paper_challenging_2025-05-19_11-45-51_seed_2",
        "gpt4t_paper_challenging_2025-05-19_12-32-10_seed_3",
        "gpt4t_paper_challenging_2025-05-19_13-24-07_seed_4",
        "gpt4t_paper_challenging_2025-05-19_13-57-18_seed_5",
    ]

    # Runs against Goal-Setting Assistant
    # chatbot_id = "study_goal_assistant"

    # experiment = "paper"
    # run_ids = [
    #     "paper_standard_2025-05-19_11-01-55_seed_1",
    #     "paper_standard_2025-05-19_11-40-28_seed_2",
    #     "paper_standard_2025-05-19_12-13-27_seed_3",
    #     "paper_standard_2025-05-19_13-23-40_seed_4",
    #     "paper_standard_2025-05-19_13-56-44_seed_5",
    #     "paper_challenging_2025-05-19_11-01-57_seed_1",
    #     "paper_challenging_2025-05-19_11-40-34_seed_2",
    #     "paper_challenging_2025-05-19_12-13-34_seed_3",
    #     "paper_challenging_2025-05-19_13-23-47_seed_4",
    #     "paper_challenging_2025-05-19_13-56-50_seed_5",
    # ]

    chatbot_dir = f"../chatbots/{chatbot_id}"
    try:
        configurator = RunConfigurator(chatbot_dir)

        # Create a list to store statistics for each run
        simulation_stats_list = []
        breakdown_and_rating_stats_list = []

        # Get statistics for each run
        for run_id in run_ids:
            try:
                stats = configurator.get_dialogue_statistics(run_id)
                breakdown_and_rating_stats = (
                    configurator.get_breakdown_and_rating_stats(run_id)
                )
                # Add run_id to the stats dictionary
                stats["run_id"] = run_id
                simulation_stats_list.append(stats)
                breakdown_and_rating_stats["run_id"] = run_id
                breakdown_and_rating_stats_list.append(breakdown_and_rating_stats)
            except Exception as e:
                print(f"Error processing run {run_id}: {e}")

        # Convert to DataFrame
        simulation_stats_df = pd.DataFrame(simulation_stats_list)
        breakdown_and_rating_stats_df = pd.DataFrame(breakdown_and_rating_stats_list)

        # Reorder columns to put run_id first
        cols = ["run_id"] + [
            col for col in simulation_stats_df.columns if col != "run_id"
        ]
        simulation_stats_df = simulation_stats_df[cols]
        cols = ["run_id"] + [
            col for col in breakdown_and_rating_stats_df.columns if col != "run_id"
        ]
        breakdown_and_rating_stats_df = breakdown_and_rating_stats_df[cols]

        # Round numeric columns to 2 decimal places
        numeric_cols = simulation_stats_df.select_dtypes(include=["float64"]).columns
        simulation_stats_df[numeric_cols] = simulation_stats_df[numeric_cols].round(2)
        numeric_cols = breakdown_and_rating_stats_df.select_dtypes(
            include=["float64"]
        ).columns
        breakdown_and_rating_stats_df[numeric_cols] = breakdown_and_rating_stats_df[
            numeric_cols
        ].round(2)

        # Print the table
        print("\nChatbot Statistics:")
        simulation_stat_headers = [
            "Run ID",
            "#UT",
            "#ST",
            "Avg ST/D",
            "Med UT Len",
            "Med CT Len",
            "User MTLD",
            "System MTLD",
        ]
        print(
            tabulate(
                simulation_stats_df,
                headers=simulation_stat_headers,
                tablefmt="grid",
                showindex=False,
            )
        )
        # Write the table to a text file
        output_file = f"{chatbot_id}_{experiment}_simulation_statistics.txt"
        output_path = output_file
        with open(output_path, "w") as f:
            f.write("Chatbot Statistics:\n")
            f.write(
                tabulate(
                    simulation_stats_df,
                    headers=simulation_stat_headers,
                    tablefmt="grid",
                    showindex=False,
                )
            )

        print(f"\nTable written to {output_path}")

        print("\nBreakdown and Rating Statistics:")
        breakdown_and_rating_stat_headers = [
            "Run ID",
            "#D with B",
            "#B",
            "B/ST",
            "#Unique B",
            "Avg. TtB",
            "Avg. Rating",
            "#Crash",
        ]
        print(
            tabulate(
                breakdown_and_rating_stats_df,
                headers=breakdown_and_rating_stat_headers,
                tablefmt="grid",
                showindex=False,
            )
        )
        # Write the table to a text file
        output_file = f"{chatbot_id}_{experiment}_breakdown_and_rating_statistics.txt"
        output_path = output_file
        with open(output_path, "w") as f:
            f.write("Breakdown and Rating Statistics:\n")
            f.write(
                tabulate(
                    breakdown_and_rating_stats_df,
                    headers=breakdown_and_rating_stat_headers,
                    tablefmt="grid",
                    showindex=False,
                )
            )

        # Compute averages and standard deviations for each simulator type
        def extract_simulator_type(run_id):
            if "autotod_multiwoz" in run_id:
                return "autotod_multiwoz"
            elif "standard" in run_id:
                return "standard"
            elif "challenging" in run_id:
                return "challenging"
            elif "testers" in run_id:
                return "testers"
            return "unknown"

        # Add simulator type column to both dataframes
        simulation_stats_df["simulator_type"] = simulation_stats_df["run_id"].apply(
            extract_simulator_type
        )
        breakdown_and_rating_stats_df["simulator_type"] = breakdown_and_rating_stats_df[
            "run_id"
        ].apply(extract_simulator_type)

        # Compute statistics for simulation stats
        numeric_cols_sim = simulation_stats_df.select_dtypes(
            include=["float64", "int64"]
        ).columns
        simulation_stats_agg = (
            simulation_stats_df.groupby("simulator_type")[numeric_cols_sim]
            .agg(["mean", "std"])
            .round(2)
        )

        # Compute statistics for breakdown and rating stats
        numeric_cols_break = breakdown_and_rating_stats_df.select_dtypes(
            include=["float64", "int64"]
        ).columns
        breakdown_stats_agg = (
            breakdown_and_rating_stats_df.groupby("simulator_type")[numeric_cols_break]
            .agg(["mean", "std"])
            .round(2)
        )

        # Reorder simulator types
        simulator_type_order = [
            "autotod_multiwoz",
            "standard",
            "challenging",
            "testers",
        ]
        simulation_stats_agg = simulation_stats_agg.reindex(simulator_type_order)
        breakdown_stats_agg = breakdown_stats_agg.reindex(simulator_type_order)

        # Print aggregated statistics
        print("\nAggregated Simulation Statistics by Simulator Type:")
        print(tabulate(simulation_stats_agg, headers="keys", tablefmt="grid"))

        print("\nAggregated Breakdown and Rating Statistics by Simulator Type:")
        print(tabulate(breakdown_stats_agg, headers="keys", tablefmt="grid"))

        # Write aggregated statistics to files
        agg_simulation_file = (
            f"{chatbot_id}_{experiment}_aggregated_simulation_statistics.txt"
        )
        with open(agg_simulation_file, "w") as f:
            f.write("Aggregated Simulation Statistics by Simulator Type:\n")
            f.write(tabulate(simulation_stats_agg, headers="keys", tablefmt="grid"))

        agg_breakdown_file = (
            f"{chatbot_id}_{experiment}_aggregated_breakdown_statistics.txt"
        )
        with open(agg_breakdown_file, "w") as f:
            f.write("Aggregated Breakdown and Rating Statistics by Simulator Type:\n")
            f.write(tabulate(breakdown_stats_agg, headers="keys", tablefmt="grid"))

        print(
            f"\nAggregated statistics written to {agg_simulation_file} and {agg_breakdown_file}"
        )

    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
