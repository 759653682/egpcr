import warnings

from src.config import DEVICE, GLOBAL_SEED
from src.pipeline import evaluate_model

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def print_results(avg_far, avg_fdr, all_faults):
    print("\n\n" + "=" * 50)
    print("--- Final Detailed Evaluation Results (WWTP Custom Dataset EGPCR) ---")
    print("=" * 50)
    print(f"{'FaultID':<10} | {'FAR':<10} | {'FDR':<10}")
    print("-" * 35)

    if all_faults:
        for fault_id, metrics in sorted(all_faults.items()):
            print(
                f"{f'ID {fault_id}':<10} | "
                f"{metrics['far']:.4f}     | {metrics['fdr']:.4f}"
            )
        print("-" * 35)
        print(f"{'Average':<10} | {avg_far:.4f}     | {avg_fdr:.4f}")
    else:
        print("No fault results were computed.")

    print("=" * 50)
    print("\n--- Script Finished ---")


def main():
    print(f"Using device: {DEVICE}")
    print("Running directly with predefined parameters...")

    final_seed = GLOBAL_SEED + 11
    avg_far, avg_fdr, all_faults = evaluate_model(final_seed)
    print_results(avg_far, avg_fdr, all_faults)


if __name__ == "__main__":
    main()
