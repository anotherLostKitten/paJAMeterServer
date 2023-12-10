sequential: sequential_classifier.c load_data.c
	gcc -o sequential sequential_classifier.c load_data.c -lm -g

clean:
	rm -rf *~* a.out sequential *.jxe
