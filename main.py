from git import Repo
from pathlib import Path
import ollama 
import chromadb
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",  # images
    ".mp4", ".mp3", ".wav",                            # media
    ".zip", ".tar", ".gz",                             # archives
    ".pdf", ".docx",                                   # docs
    ".lock",                                           # lock files
    ".pyc", ".pyo",                                    # python cache
    ".exe", ".bin", ".so", ".dylib",                   # binaries
    ".gitignore"
}

# folders to skip entirely
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__",
    ".venv", "venv", "env",
    "dist", "build", ".next",
    ".idea", ".vscode",
}
class GitHubHelper:

    def __init__(self):
        self.repo=None
        self.client = chromadb.PersistentClient(path="./chroma-store")
        self.collection = self.client.create_collection(name="docs")
    
    def clone_repository(self,remote_url,target_dir):
        self.repo=Repo.clone_from(remote_url,target_dir)
        assert not self.repo.bare
        print(f"Cloned Successfull to {target_dir}")
        
    def walkRepo(self,target_dir):
        root_dir=Path(target_dir)
        files=[]
        for file_path in root_dir.rglob("*"):
            if(file_path.is_file()):
                print(f"Reading :{file_path}")
                try: 

                    if any(part in SKIP_DIRS for part in file_path.parts):
                        continue
                    if file_path.suffix.lower() in SKIP_EXTENSIONS:
                        continue
                    content=file_path.read_text(encoding='utf-8')
                    if not content.strip():
                        continue
                    files.append({
                    "path": str(file_path.relative_to(root_dir)),
                    "content": content,
                    "extension": file_path.suffix.lower(),
                    "size": file_path.stat().st_size,
                })

                except Exception as e:
                    print(f"Could not read {file_path}:{e}")
        return files
    
    def dbStore(self,files):
        print("Embedding and storing in ChromaDB...")
        for i,file in enumerate(files):
            print(f"Embedding {file['path']}...")
            response=ollama.embed(model='nomic-embed-text',input=file["content"])
            self.collection.add(
                ids=[str(i)],
                embeddings=[response["embeddings"][0]],
                documents=[file["content"]],
                metadatas=[{
                    "path": file["path"],
                    "extension": file["extension"],
                }]
            )

        print(f"Stored {len(files)} files in ChromaDB.")
    
    def chunkFiles(self,files):
        EXTENSION_MAP = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".jsx": Language.JS,
        ".tsx": Language.TS,
        ".go": Language.GO,
        ".java": Language.JAVA,
        ".rb": Language.RUBY,
        ".rs": Language.RUST,
        ".cpp": Language.CPP,
        ".c": Language.C,
        ".cs": Language.CSHARP,
        ".md": Language.MARKDOWN,
        ".html": Language.HTML,
        }
        all_chunks=[]
        for file in files:
            ext=file['extension']
            language=EXTENSION_MAP.get(ext)
            if language:
                splitter=RecursiveCharacterTextSplitter.from_language(
                    language=language,
                    chunk_size=1000,
                    chunk_overlap=100,
                )
            else:
                splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                )
            chunks=splitter.split_text(file["content"])
            for chunk in chunks:
                all_chunks.append({
                    "content":chunk,
                    "path":file["path"],
                    "extension":ext,
                })
        return all_chunks



helper=GitHubHelper()
helper.clone_repository("https://github.com/KrishnaKalra/CSE-Chapter-28-server.git","./cloned-repos")
files=helper.walkRepo("./cloned-repos")
files=helper.chunkFiles(files)
helper.dbStore(files)

#print(files)