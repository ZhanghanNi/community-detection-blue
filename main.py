import networkx as nx
import sys

sys.path.append("Util")
sys.path.append("Aidan/girvan_newman_implementation")
sys.path.append("Yang_Tan")
sys.path.append("Tony/PPI")
import girvan_newman_implementation as gn
import louvain_method_implementation as lm
import run_eval


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 main.py <algorithm> <dataset>")
        print("Where <algorithm> is 'girvan_newman' or 'louvain_method'")
        print("Where <dataset> is 'karate_club' or INSERT OTHER DATASETS HERE")
        sys.exit(1)

    algorithm = sys.argv[1]

    if algorithm not in ["girvan_newman", "louvain_method"]:
        print(f"Error: '{algorithm}' is not a valid algorithm.")
        print("Valid options are 'girvan_newman' or 'louvain_method'.")
        sys.exit(1)

    print(f"Selected algorithm: {algorithm}")

    dataset = sys.argv[2]

    if dataset not in ["karate_club"]:
        print(f"Error: '{algorithm}' is not a valid dataset.")
        print("Valid options are 'karate_club' or INSERT OTHER DATASETS HERE.")
        sys.exit(1)

    karate_club_graph = nx.karate_club_graph()

    match algorithm:
        case "girvan_newman":
            match dataset:
                case "karate_club":
                    label = "Karate Club"
                    gn.main(karate_club_graph, weighted=True, dataset_name=label)
                    gn.main(karate_club_graph, weighted=False, dataset_name=label)
        case "louvain_method":
            match dataset:
                case "karate_club":
                    lm.main(karate_club_graph, dataset_name="Karate Club")

    run_eval.main()


if __name__ == "__main__":
    main()
