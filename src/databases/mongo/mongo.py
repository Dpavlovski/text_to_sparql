import logging
from typing import Optional, Any, List, Dict, TypeVar
from typing import Type as TypingType

from bson import ObjectId
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError


class MongoEntry(BaseModel):
    id: Optional[ObjectId] = None

    class Config:
        arbitrary_types_allowed = True


T = TypeVar('T', bound=MongoEntry)


class MongoDBDatabase:
    client: MongoClient
    db: Any

    def __init__(self):
        self.client = MongoClient(f"mongodb://root:example@localhost:27017/")
        self.db = self.client["text_to_sparql"]

    def add_entry(
            self,
            entity: BaseModel,
            collection_name: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        collection_name = entity.__class__.__name__ if collection_name is None else collection_name
        collection = self.db[collection_name]
        entry = entity.model_dump()
        if "id" in list(entry.keys()):
            entry.pop("id")
        if metadata:
            entry.update(metadata)

        collection.insert_one(entry)
        return True

    def add_entry_dict(
            self,
            entity: Dict[str, Any],
            collection_name: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        collection = self.db[collection_name]
        entry = entity
        if "id" in list(entry.keys()):
            entry.pop("id")
        if metadata:
            entry.update(metadata)

        collection.insert_one(entry)
        return True

    def get_entries(
            self,
            class_type: TypingType[T],
            doc_filter: Dict[str, Any] = None,
            collection_name: Optional[str] = None,
    ) -> List[T]:
        collection_name = class_type.__name__ if collection_name is None else collection_name
        collection = self.db[collection_name]
        documents = collection.find(doc_filter or {})

        class_fields = class_type.model_fields.keys()

        results = []
        for doc in documents:
            entry_attr = {}

            for field in class_fields:
                if field == "id":
                    entry_attr[field] = doc["_id"]
                else:
                    entry_attr[field] = doc[field] if field in doc else None

            entry = class_type(**entry_attr)
            results.append(entry)

        return results

    def get_entries_dict(
            self,
            collection_name: str,
            doc_filter: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        collection = self.db[collection_name]
        documents = collection.find(doc_filter or {})

        results = []
        for doc in documents:
            doc['id'] = doc.pop('_id')
            results.append(doc)

        return results

    def set_unique_index(self, collection_name: str, field_name: str):
        try:
            collection = self.db[collection_name]
            collection.create_index([(field_name, 1)], unique=True)
            logging.info(f"Unique index set for '{field_name}' in '{collection_name}' collection.")
        except DuplicateKeyError:
            logging.info(f"Cannot create unique index on '{field_name}' due to duplicate values.")
        except Exception as e:
            logging.info(f"An error occurred: {e}")

    def get_ids(
            self,
            class_type: TypingType[BaseModel],
            collection_name: Optional[str] = None,
            doc_filter: Dict[str, Any] = None,
    ) -> List[ObjectId]:
        collection_name = class_type.__name__ if collection_name is None else collection_name
        collection = self.db[collection_name]

        ids_cursor = collection.find(doc_filter or {}, {"_id": 1})

        return [doc["_id"] for doc in ids_cursor]

    def get_entity(
            self,
            id: ObjectId,
            class_type: TypingType[T],
            collection_name: Optional[str] = None,
    ) -> Optional[T]:
        collection_name = class_type.__name__ if collection_name is None else collection_name
        collection = self.db[collection_name]

        document = collection.find_one({"_id": id})
        if document:
            class_fields = class_type.model_fields.keys()
            filtered_doc = {key: value for key, value in document.items() if key in class_fields}
            filtered_doc["_id"] = id

            instance = class_type(**filtered_doc)
            return instance

        return None

    def update_entity(
            self,
            entity: MongoEntry,
            collection_name: Optional[str] = None,
            update: Optional[Dict[str, Any]] = None
    ) -> bool:
        collection_name = entity.__class__.__name__ if collection_name is None else collection_name
        collection = self.db[collection_name]

        entity_dict = entity.model_dump()

        if update:
            entity_dict.update(update)

        result = collection.update_one(
            {"_id": entity.id},
            {"$set": entity_dict}
        )

        return result.modified_count > 0

    def update_entity_dict(
            self,
            entity: Dict[str, Any],
            collection_name: Optional[str] = None,
            update: Optional[Dict[str, Any]] = None
    ) -> bool:
        collection_name = entity.__class__.__name__ if collection_name is None else collection_name
        collection = self.db[collection_name]

        entity_dict = entity

        if update:
            entity_dict.update(update)

        result = collection.update_one(
            {"_id": entity["id"]},
            {"$set": entity_dict}
        )

        return result.modified_count > 0

    def delete_collection(self, collection_name: str) -> bool:
        if collection_name not in self.db.list_collection_names():
            logging.info(f"Collection '{collection_name}' does not exist.")

        self.db[collection_name].drop()
        return True

    def delete_entity(
            self,
            entity: TypingType[T],
            collection_name: Optional[str] = None
    ) -> bool:
        collection_name = entity.__class__.__name__ if collection_name is None else collection_name
        collection = self.db[collection_name]

        result = collection.delete_one({"_id": entity.id})

        return result.deleted_count > 0

    def delete_entity_dict(
            self,
            entity_dict: Dict[str, Any],
            collection_name: str
    ) -> bool:
        collection_name = collection_name
        collection = self.db[collection_name]

        result = collection.delete_one({"_id": entity_dict["id"]})

        return result.deleted_count > 0
