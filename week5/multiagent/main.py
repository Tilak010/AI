import sys

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    except Exception:
        pass

from single_agent import run_single_agent
from multi_agent import run_multi_agent
from compare import run_benchmark

def print_menu():
    print("\n" + "=" * 65)
    print("🤖 AI AGENT ARCHITECTURE & TOKEN CONSUMPTION BENCHMARK")
    print("=" * 65)
    print(" 1. Run Single Agent   (1 LLM handling Search + Math)")
    print(" 2. Run Multi-Agent    (Manager + Search Agent + Math Agent)")
    print(" 3. Run Benchmark      (Side-by-side token comparison)")
    print(" 4. Custom Task Query  (Run benchmark with your own prompt)")
    print(" 5. Exit")
    print("=" * 65)

DEFAULT_TASK = (
    "Search for the current population of Tokyo and the population of New York City, "
    "and calculate the difference between their populations and the percentage by which Tokyo is larger."
)

def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["--single", "-s"]:
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else DEFAULT_TASK
            run_single_agent(query)
            return
        elif arg in ["--multi", "-m"]:
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else DEFAULT_TASK
            run_multi_agent(query)
            return
        elif arg in ["--compare", "-c", "--benchmark", "-b"]:
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else DEFAULT_TASK
            run_benchmark(query)
            return

    while True:
        print_menu()
        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            custom = input(f"\nEnter task (press Enter for default):\n[{DEFAULT_TASK}]\n> ").strip()
            task = custom if custom else DEFAULT_TASK
            run_single_agent(task)
        elif choice == "2":
            custom = input(f"\nEnter task (press Enter for default):\n[{DEFAULT_TASK}]\n> ").strip()
            task = custom if custom else DEFAULT_TASK
            run_multi_agent(task)
        elif choice == "3":
            run_benchmark(DEFAULT_TASK)
        elif choice == "4":
            custom = input("\nEnter your custom prompt:\n> ").strip()
            if custom:
                run_benchmark(custom)
            else:
                print("Prompt cannot be empty.")
        elif choice in ["5", "exit", "quit", "q"]:
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
