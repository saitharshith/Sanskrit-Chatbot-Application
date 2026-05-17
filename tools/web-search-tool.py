from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

@tool
def search_grammar_and_philosophy_web(query: str) -> str:
    """
    Use this tool ONLY when the local database does not have enough deep philosophical
    meaning, or when looking up advanced Paninian grammar, root words, or Mahabharata lore.
    """
    search = DuckDuckGoSearchResults()

    trusted_domains = (
        "site:ashtadhyayi.com OR "
        "site:sanskritdocuments.org OR "
        "site:wisdomlib.org OR "
        "site:gitasupersite.iitk.ac.in OR "
        "site:sacred-texts.com"
    )

    strict_query = f"{query} {trusted_domains}"

    try:
        results = search.run(strict_query)

        if not results:
            return "The ancient web texts did not yield a result. Try simplifying the search term."

        return f"Authoritative Web Search Excerpts: {results}"

    except Exception as e:
        return f"My vision of the external texts is clouded: {str(e)}"