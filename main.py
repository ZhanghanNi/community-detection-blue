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
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <algorithm>")
        print("Where <algorithm> is 'girvan_newman' or 'louvain_method'")
        sys.exit(1)

    algorithm = sys.argv[1]

    if algorithm not in ["girvan_newman", "louvain_method"]:
        print(f"Error: '{algorithm}' is not a valid algorithm.")
        print("Valid options are 'girvan_newman' or 'louvain_method'.")
        sys.exit(1)

    print(f"Selected algorithm: {algorithm}")

    match algorithm:
        case "girvan_newman":
            print("calling girvan-newman")
            gn.main(weighted=True)
            gn.main(weighted=False)
        case "louvain_method":
            lm.main()
    
    run_eval.main()
    

if __name__ == "__main__":
    main()
