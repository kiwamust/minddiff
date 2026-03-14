"""SQLAlchemy ORM models."""

import json
import secrets

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    cycle_interval = Column(Integer, default=7)  # days
    created_at = Column(DateTime, default=func.now())

    members = relationship("Member", back_populates="team", cascade="all, delete-orphan")
    cycles = relationship("InputCycle", back_populates="team", cascade="all, delete-orphan")


class Member(Base):
    __tablename__ = "members"

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    display_name = Column(String(100), nullable=False)
    email = Column(String(254), nullable=False)
    role = Column(String(20), default="member")  # pm | member
    token = Column(String(64), unique=True, default=lambda: secrets.token_urlsafe(32))
    created_at = Column(DateTime, default=func.now())

    team = relationship("Team", back_populates="members")
    responses = relationship("Response", back_populates="member", cascade="all, delete-orphan")


class InputCycle(Base):
    __tablename__ = "input_cycles"

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    cycle_number = Column(Integer, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    status = Column(String(20), default="open")  # open | closed | reported
    created_at = Column(DateTime, default=func.now())

    team = relationship("Team", back_populates="cycles")
    responses = relationship("Response", back_populates="cycle", cascade="all, delete-orphan")
    report = relationship(
        "Report", back_populates="cycle", uselist=False, cascade="all, delete-orphan"
    )


class Response(Base):
    __tablename__ = "responses"

    id = Column(Integer, primary_key=True)
    input_cycle_id = Column(Integer, ForeignKey("input_cycles.id"), nullable=False)
    member_id = Column(Integer, ForeignKey("members.id"), nullable=False)
    dimension = Column(Integer, nullable=False)  # 1-5
    content = Column(Text, nullable=False, default="")
    is_draft = Column(Boolean, default=True)
    submitted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    cycle = relationship("InputCycle", back_populates="responses")
    member = relationship("Member", back_populates="responses")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True)
    input_cycle_id = Column(Integer, ForeignKey("input_cycles.id"), nullable=False, unique=True)
    synthesis = Column(Text, default="{}")  # JSON
    divergences = Column(Text, default="{}")  # JSON
    alignment_scores = Column(Text, default="{}")  # JSON
    raw_llm_response = Column(Text, default="")
    generated_at = Column(DateTime, default=func.now())

    cycle = relationship("InputCycle", back_populates="report")

    def get_synthesis(self) -> dict:
        return json.loads(self.synthesis) if self.synthesis else {}

    def get_divergences(self) -> dict:
        return json.loads(self.divergences) if self.divergences else {}

    def get_alignment_scores(self) -> dict:
        return json.loads(self.alignment_scores) if self.alignment_scores else {}
