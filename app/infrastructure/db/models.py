from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class InstituteModel(Base):
    __tablename__ = 'institutes'
    code = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    departments = relationship('DepartmentModel', back_populates='institute')


class DepartmentModel(Base):
    __tablename__ = 'departments'
    code = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    institute_code = Column(String, ForeignKey('institutes.code'), nullable=False)
    institute = relationship('InstituteModel', back_populates='departments')
    programs = relationship('ProgramModel', back_populates='department')


class ProgramModel(Base):
    __tablename__ = 'programs'
    code = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    department_code = Column(String, ForeignKey('departments.code'), nullable=False)
    is_ino = Column(Boolean, default=False, nullable=False)
    is_international = Column(Boolean, default=False, nullable=False)
    department = relationship('DepartmentModel', back_populates='programs')


class SubmissionStatsModel(Base):
    __tablename__ = 'submission_stats'
    program_code = Column(String, ForeignKey('programs.code'), primary_key=True)
    num_places = Column(Integer, nullable=False)
    num_applications = Column(Integer, nullable=False)
    generated_at = Column(DateTime, nullable=False)
    program = relationship('ProgramModel', back_populates='stats')


ProgramModel.stats = relationship('SubmissionStatsModel', uselist=False, back_populates='program')


class ApplicantModel(Base):
    __tablename__ = 'applicants'
    id = Column(String, primary_key=True)


class ApplicationModel(Base):
    __tablename__ = 'applications'
    # composite PK: program + applicant
    program_code = Column(String, ForeignKey('programs.code'), primary_key=True)
    applicant_id = Column(String, ForeignKey('applicants.id'), primary_key=True)
    total_score = Column(Integer, nullable=False)
    vi_score = Column(Integer, nullable=False)
    subject1_score = Column(Integer, nullable=False)
    subject2_score = Column(Integer, nullable=False)
    id_achievements = Column(Integer, nullable=False)
    target_id_achievements = Column(Integer, nullable=False)
    priority = Column(Integer, nullable=False)
    consent = Column(Boolean, nullable=False)
    review_status = Column(String, nullable=False)

    applicant = relationship('ApplicantModel')
    program = relationship('ProgramModel')
