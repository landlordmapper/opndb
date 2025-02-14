from typing import TypedDict


class AddressColObj(TypedDict):
    complete_addr: str
    street: str | None
    city: str | None
    state: str | None
    zip: str | None