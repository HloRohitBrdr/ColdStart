from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, select
import uuid
import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

DATABASE_URL = "sqlite+aiosqlite:///./coldstart.db"

# Setup DB
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# FastAPI app
app = FastAPI()
router = APIRouter(prefix="/api")

# Add CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB Models ---
class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    name = Column(String)
    email = Column(String)
    location = Column(String)
    age = Column(Integer)
    gender = Column(String)
    interests = Column(String)

class ProductDB(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String, unique=True, index=True)
    name = Column(String)
    category = Column(String)
    price = Column(Float)

# --- Pydantic Models ---
class UserCreate(BaseModel):
    name: str
    email: str
    location: str
    age: int
    gender: str
    interests: str

class ProductCreate(BaseModel):
    name: str
    category: str
    price: float

class Recommendation(BaseModel):
    product_id: str
    product_name: str
    category: str
    price: float
    score: float
    reasoning: str

# --- Recommendation Engine ---
class ColdStartEngine:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words="english")

    def content_similarity(self, user_interest, categories):
        if not user_interest or not categories:
            return np.zeros(len(categories))
        corpus = [user_interest] + categories
        tfidf_matrix = self.tfidf.fit_transform(corpus)
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    def demo_similarity(self, user, others):
        sims = []
        for o in others:
            age_score = max(0, 1 - abs(user['age'] - o['age']) / 50)
            gender_score = 1 if user['gender'].lower() == o['gender'].lower() else 0.3
            location_score = 1 if user['location'].lower() == o['location'].lower() else 0.2
            sims.append(0.4 * age_score + 0.3 * gender_score + 0.3 * location_score)
        return np.mean(sims) if sims else 0.5

    def price_score(self, age, price, all_prices):
        mean, std = np.mean(all_prices), np.std(all_prices)
        if age < 25: pref = mean * 0.7
        elif age < 35: pref = mean
        else: pref = mean * 1.3
        return math.exp(-abs(price - pref) / (2 * std))

    def popularity_score(self, category, all_products):
        freq = sum(p['category'] == category for p in all_products)
        return freq / len(all_products) if all_products else 0.5

    def recommend(self, user, users, products, top_k=5):
        categories = [p['category'] for p in products]
        prices = [p['price'] for p in products]
        content = self.content_similarity(user['interests'], categories)
        demo = self.demo_similarity(user, users)

        recs = []
        for i, p in enumerate(products):
            cscore = content[i]
            pscore = self.price_score(user['age'], p['price'], prices)
            popscore = self.popularity_score(p['category'], products)
            final = 0.4 * cscore + 0.3 * pscore + 0.2 * popscore + 0.1 * demo
            recs.append({
                "product_id": p['product_id'], "product_name": p['name'],
                "category": p['category'], "price": p['price'],
                "score": final,
                "reasoning": f"Content={cscore:.2f}, Price={pscore:.2f}, Pop={popscore:.2f}, Demo={demo:.2f}"
            })
        return sorted(recs, key=lambda x: x['score'], reverse=True)[:top_k]

engine_instance = ColdStartEngine()

# --- API Endpoints ---
@router.post("/users")
async def create_user(user: UserCreate):
    uid = str(uuid.uuid4())
    async with SessionLocal() as db:
        db_user = UserDB(user_id=uid, **user.dict())
        db.add(db_user)
        await db.commit()
        return {"user_id": uid, **user.dict()}

@router.post("/products")
async def create_product(product: ProductCreate):
    pid = str(uuid.uuid4())
    async with SessionLocal() as db:
        db_prod = ProductDB(product_id=pid, **product.dict())
        db.add(db_prod)
        await db.commit()
        return {"product_id": pid, **product.dict()}

@router.get("/users")
async def get_users():
    async with SessionLocal() as db:
        users = await db.execute(select(UserDB))
        return [u.__dict__ for u in users.scalars()]

@router.post("/sample-data")
async def create_sample_data():
    sample_users = [
        UserCreate(name="Riya", email="riya@example.com", location="Delhi", age=22, gender="Female", interests="books, fashion, makeup"),
        UserCreate(name="Aman", email="aman@example.com", location="Mumbai", age=30, gender="Male", interests="electronics, gadgets, games"),
    ]
    sample_products = [
        ProductCreate(name="Book Lamp", category="Books", price=199.0),
        ProductCreate(name="Gaming Console", category="Electronics", price=299.0),
        ProductCreate(name="Lipstick Set", category="Fashion", price=99.0),
        ProductCreate(name="Smartwatch", category="Electronics", price=399.0),
        ProductCreate(name="Yoga Mat", category="Fitness", price=149.0),
    ]
    async with SessionLocal() as db:
        for u in sample_users:
            db_user = UserDB(user_id=str(uuid.uuid4()), **u.dict())
            db.add(db_user)
        for p in sample_products:
            db_product = ProductDB(product_id=str(uuid.uuid4()), **p.dict())
            db.add(db_product)
        await db.commit()
    return {"message": "Sample data created"}

@router.get("/recommendations/{user_id}")
async def recommend(user_id: str, top_k: int = 5):
    async with SessionLocal() as db:
        user = await db.execute(select(UserDB).where(UserDB.user_id == user_id))
        user = user.scalar()
        if not user:
            raise HTTPException(404, "User not found")

        users_raw = await db.execute(select(UserDB))
        users = [u.__dict__ for u in users_raw.scalars() if u.user_id != user_id]

        products_raw = await db.execute(select(ProductDB))
        products = [p.__dict__ for p in products_raw.scalars()]

        recs = engine_instance.recommend(user.__dict__, users, products, top_k)
        return {"user_id": user_id, "recommendations": recs}

# Register router
app.include_router(router)

@app.on_event("startup")
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Mount static folder for frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")