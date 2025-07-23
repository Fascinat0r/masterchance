from sqlalchemy import Column, String, Boolean, ForeignKey
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
