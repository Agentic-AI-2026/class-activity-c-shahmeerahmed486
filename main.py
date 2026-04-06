import asyncio
import sys

from graph import run_agent


async def _main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your query: ").strip()

    if not query:
        print("No query provided.")
        return

    result = await run_agent(query)

    print("\nFinal Answer:\n")
    print(result.get("final_answer", ""))

    print("\nSteps:\n")
    for i, step in enumerate(result.get("steps", []), start=1):
        print(f"{i}. {step}")


if __name__ == "__main__":
    asyncio.run(_main())
