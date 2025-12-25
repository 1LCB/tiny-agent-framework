import os

class Skill:
    """
    A default, filesystem-based implementation of a skill.

    This class represents one possible way to package and load skills using
    files (e.g. a SKILL.md file plus optional resources on disk). It is not a
    required or canonical representation of a skill.
    """
    def __init__(self, name: str, description: str, metadata: dict, file_path: str, resources: list[str]):
        self.name = name
        self.description = description
        self.metadata = metadata
        self.file_path = file_path
        self.resources = resources  # relative paths

        self.loaded = False
        self._body = None

    def load(self) -> str:
        if self.loaded:
            return self._body

        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        parts = content.split("---", 2)
        self._body = parts[2].lstrip() if len(parts) == 3 else ""
        self.loaded = True
        return self._body

    def load_resource(self, resource: str) -> str:
        base_dir = os.path.dirname(self.file_path)
        resource_path = os.path.join(base_dir, resource)

        with open(resource_path, "r", encoding="utf-8") as f:
            return f.read()

    @classmethod
    def from_folder(cls, folder_path: str) -> list["Skill"]:
        skills = []

        for entry in os.listdir(folder_path):
            skill_dir = os.path.join(folder_path, entry)
            if not os.path.isdir(skill_dir):
                continue

            skill_file = os.path.join(skill_dir, "SKILL.md")
            if os.path.isfile(skill_file):
                skills.append(cls.from_file(skill_file))

        return skills

    @classmethod
    def from_file(cls, file_path: str) -> "Skill":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.startswith("---"):
            raise ValueError("Missing YAML frontmatter")

        frontmatter = content.split("---", 2)[1]
        data = cls.__parse_frontmatter(frontmatter)

        name = data.get("name")
        description = data.get("description")

        if not name or not description:
            raise ValueError("Skill must define name and description")

        metadata = data.get("metadata", {})

        resources = cls.__discover_resources(file_path)

        return cls(name, description, metadata, file_path, resources)

    @staticmethod
    def __discover_resources(skill_file: str) -> list[str]:
        base_dir = os.path.dirname(skill_file)
        resources = []

        for root, _, files in os.walk(base_dir):
            for file in files:
                if file == "SKILL.md":
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                resources.append(rel_path)

        return resources

    @staticmethod
    def __parse_frontmatter(text: str) -> dict:
        result = {}
        current_key = None

        for line in text.splitlines():
            if not line.strip():
                continue

            if ":" in line and not line.startswith(" "):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if value == "":
                    result[key] = {}
                    current_key = key
                else:
                    result[key] = value.strip('"')
                    current_key = None
            elif current_key:
                subkey, subvalue = line.strip().split(":", 1)
                result[current_key][subkey.strip()] = subvalue.strip().strip('"')

        return result

