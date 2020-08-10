all: crossValidationCL.exe

iOCL=-I. -I/opt/AMDAPP/3.0/include
lOCL=-L/opt/AMDAPP/3.0/lib -lOpenCL -g

CC=g++
CFLAGS=-Wall -O2 -std=c++11 $(iOCL) $(lOCL) 

OBJS=kNN.o distance.o 
CVOBJS=evaluation.o YoUtil.o kNN.o 

%.o: src/%.cpp inc/%.h
	$(CC) $(CFLAGS) -c $< -o $@

YoUtil.o: src/YoUtil.cpp inc/YoUtil.hpp
	$(CC) $(CFLAGS) -c $< -o $@

crossValidationCL.exe: crossValidationCL.cpp $(CVOBJS)
	$(CC) $(CFLAGS) $< -o $@ $(CVOBJS) -lOpenCL

test: main.exe
	valgrind ./main.exe data/01_kid.csv data/01_questions.csv 3 3 2 0 CR 
	valgrind ./main.exe data/01_kid.csv data/01_questions.csv 3 3 2 1 CR 
	valgrind ./main.exe data/01_kid.csv data/01_questions.csv 3 3 2 2 CR 
	valgrind ./main.exe data/01_kid.csv data/01_questions.csv 3 3 2 3 CR 1.5
	valgrind ./main.exe data/01_kid.csv data/01_questions.csv 3 3 2 4 CR 
	valgrind ./main.exe data/02_iris.csv data/02_iris_questions.csv 5 4 1 0 C 
	valgrind ./main.exe data/02_iris.csv data/02_iris_questions.csv 5 4 1 1 C 
	valgrind ./main.exe data/02_iris.csv data/02_iris_questions.csv 5 4 1 2 C 
	valgrind ./main.exe data/02_iris.csv data/02_iris_questions.csv 5 4 1 3 C 1.75
	valgrind ./main.exe data/02_iris.csv data/02_iris_questions.csv 5 4 1 4 C 

testCV: crossValidation.exe
	valgrind ./crossValidation.exe data/01_kid.csv 6 3 3 2 0 CR
	valgrind ./crossValidation.exe data/01_kid.csv 6 3 3 2 1 CR
	valgrind ./crossValidation.exe data/01_kid.csv 6 3 3 2 2 CR
	valgrind ./crossValidation.exe data/01_kid.csv 6 3 3 2 3 CR 1.5
	valgrind ./crossValidation.exe data/01_kid.csv 6 3 3 2 4 CR
#	./crossValidationCL.exe Advanced data/01_kid.csv 6 3 3 2 0 CR
01:crossValidationCL.exe
	./crossValidationCL.exe Advanced data/01_kid.csv 6 3 3 2 0 CR
	./crossValidationCL.exe Advanced data/01_kid.csv 6 3 3 2 1 CR
	./crossValidationCL.exe Advanced data/01_kid.csv 6 3 3 2 2 CR
	./crossValidationCL.exe Advanced data/01_kid.csv 6 3 3 2 4 CR
02:crossValidationCL.exe
	./crossValidationCL.exe Advanced data/02_iris.csv 6 5 4 1 0 C
	./crossValidationCL.exe Advanced data/02_iris.csv 6 5 4 1 1 C
	./crossValidationCL.exe Advanced data/02_iris.csv 6 5 4 1 2 C
	./crossValidationCL.exe Advanced data/02_iris.csv 6 5 4 1 4 C

BG:crossValidationCL.exe
	./crossValidationCL.exe Advanced data/03_Skin_NonSkin.csv 1000 5 3 1 0 C

04:crossValidationCL.exe
	./crossValidationCL.exe Advanced data/04_music_default.csv 6 5 68 2 0 RR
	./crossValidationCL.exe Advanced data/04_music_default.csv 6 5 68 2 1 RR
	./crossValidationCL.exe Advanced data/04_music_default.csv 6 5 68 2 2 RR
	./crossValidationCL.exe Advanced data/04_music_default.csv 6 5 68 2 4 RR

05:crossValidationCL.exe
	./crossValidationCL.exe Advanced data/05_music_plus.csv 6 5 116 2 0 CC
	./crossValidationCL.exe Advanced data/05_music_plus.csv 6 5 116 2 1 CC
	./crossValidationCL.exe Advanced data/05_music_plus.csv 6 5 116 2 2 CC
	./crossValidationCL.exe Advanced data/05_music_plus.csv 6 5 116 2 4 CC
clean:
	rm *.exe *.o
