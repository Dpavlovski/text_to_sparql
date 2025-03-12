from typing import Dict, List

from pywikibot import Site, Page
from tqdm import tqdm


def get_recent_changes(start: str, end: str) -> Dict[str, List[str]]:
    site = Site("wikidata", "wikidata")
    repo = site.data_repository()

    recent_changes = site.recentchanges(
        namespaces=[0],
        start=start,
        end=end,
        reverse=True,
        total=5
    )

    items: Dict[str, List[str]] = {}

    for change in tqdm(recent_changes, desc="Fetching recent changes"):
        entity_id = change['title']
        page = Page(repo, entity_id)

        try:
            if page.isRedirectPage():
                page = page.getRedirectTarget()
                print(f"{entity_id} is a redirect. Redirecting to {page.title()}")

            data = page.get()

            change_type = change["type"]
            if change_type not in items:
                items[change_type] = []

            items[change_type].append(data)

        except Exception as e:
            print(f"Error processing {entity_id}: {e}")

    return items


recent_changes_data = get_recent_changes(
    start="2025-02-19T00:00:00Z",
    end="2025-02-20T23:59:59Z"
)

for t, changes in recent_changes_data.items():
    print(f"Change Type: {t}")
    for item in changes:
        print(item)
    print("-" * 50)
