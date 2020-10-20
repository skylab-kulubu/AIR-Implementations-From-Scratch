/*
 ============================================================================
 Name        : Credit-Approval-Tool
 Author      : Berk Sudan
 Version     : 1.2
 Copyright   : No copyright, free to use, free to distribute.
 Description : Credit-Approval-Tool in C, Ansi-style
 ============================================================================
 */
#define INPUT_DATASET_NAME "input_randomized.csv"
#define NUM_OF_INST		407
#define NUM_OF_ATTRS		16
#define MAX_VALUE_SIZE		9
#define EXIT_SUCCESS 		0
#define EXIT_FAIL 		-1
#define ABS(N) 			(((N)<0)?(-(N)):(N))
#define MIN(X, Y) 		(((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) 		(((X) > (Y)) ? (X) : (Y))
#define NOM_DIST_COEFF 		300

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

struct Instance {
	int ID;
	char attr1, attr4, attr9, attr10, attr12, attr13;
	float attr2, attr3, attr8, attr11, attr14, attr15;
	char attr5[3], attr6[3], attr7[3];
	char attr16_class;
};
typedef struct Instance STR_INST;

struct DistanceVal {
	int ID;
	float dist;
};
typedef struct DistanceVal STR_DIST_VAL;

size_t my_getline(char **lineptr, size_t *n, FILE *stream) {
    char *bufptr = NULL;
    char *p = bufptr;
    size_t size;
    int c;

    if (lineptr == NULL) {
        return -1;
    }
    if (stream == NULL) {
        return -1;
    }
    if (n == NULL) {
        return -1;
    }
    bufptr = *lineptr;
    size = *n;

    c = fgetc(stream);
    if (c == EOF) {
        return -1;
    }
    if (bufptr == NULL) {
        bufptr = malloc(128);
        if (bufptr == NULL) {
            return -1;
        }
        size = 128;
    }
    p = bufptr;
    while(c != EOF) {
        if ((p - bufptr) > (size - 1)) {
            size = size + 128;
            bufptr = realloc(bufptr, size);
            if (bufptr == NULL) {
                return -1;
            }
        }
        *p++ = c;
        if (c == '\n') {
            break;
        }
        c = fgetc(stream);
    }

    *p++ = '\0';
    *lineptr = bufptr;
    *n = size;

    return p - bufptr - 1;
}


char** parseCommaSeperatedValues(char *str) {
	int i;
	char *parsedVal;
	char **parsedVals;

	parsedVals = (char**) malloc(sizeof(char*) * NUM_OF_ATTRS);
	for (i = 0; i < NUM_OF_ATTRS; i++)
		parsedVals[i] = (char*) malloc(sizeof(char) * MAX_VALUE_SIZE);

	parsedVal = strtok(str, ",");
	for (i = 0; parsedVal != NULL; i++) {
		strcpy(parsedVals[i], parsedVal);
		parsedVal = strtok(NULL, ",");
	}
	return parsedVals;
}
void ifNullExitMessage(void *ptr, char msg[64]) {
	if (ptr == NULL) {
		printf("%s\n", msg);
		exit(EXIT_FAIL);
	}
}

void printDashes() {
	printf("----------------------------------------\n");
}

void printDataSet(STR_INST *ds, int numOfInstances) {
	int i;
	for (i = 0; i < numOfInstances; i++) {
		printf("%d. <<%d,%c,%.2f,%.2f,%c,%s,", (i + 1), ds[i].ID, ds[i].attr1,
				ds[i].attr2, ds[i].attr3, ds[i].attr4, ds[i].attr5);
		printf("%s,%s,%.2f,%c,%c,%.2f,", ds[i].attr6, ds[i].attr7, ds[i].attr8,
				ds[i].attr9, ds[i].attr10, ds[i].attr11);
		printf("%c,%c,%.2f,%.2f,%c>>\n", ds[i].attr12, ds[i].attr13,
				ds[i].attr14, ds[i].attr15, ds[i].attr16_class);

	}
}

void assignStringArrToInstance(int ID, STR_INST* anInstance, char **newVals) {
	anInstance->ID = ID;
	anInstance->attr1 = newVals[0][0];
	anInstance->attr2 = atof(newVals[1]);
	anInstance->attr3 = atof(newVals[2]);
	anInstance->attr4 = newVals[3][0];
	strcpy(anInstance->attr5, newVals[4]);
	strcpy(anInstance->attr6, newVals[5]);
	strcpy(anInstance->attr7, newVals[6]);
	anInstance->attr8 = atof(newVals[7]);
	anInstance->attr9 = newVals[8][0];
	anInstance->attr10 = newVals[9][0];
	anInstance->attr11 = atof(newVals[10]);
	anInstance->attr12 = newVals[11][0];
	anInstance->attr13 = newVals[12][0];
	anInstance->attr14 = atof(newVals[13]);
	anInstance->attr15 = atof(newVals[14]);
	anInstance->attr16_class = newVals[15][0];
}

void assignInstToInst(STR_INST* inst1, STR_INST* inst2) {

	inst1->ID = inst2->ID;
	inst1->attr1 = inst2->attr1;
	inst1->attr2 = inst2->attr2;
	inst1->attr3 = inst2->attr3;
	inst1->attr4 = inst2->attr4;
	strcpy(inst1->attr5, inst2->attr5);
	strcpy(inst1->attr6, inst2->attr6);
	strcpy(inst1->attr7, inst2->attr7);
	inst1->attr8 = inst2->attr8;
	inst1->attr9 = inst2->attr9;
	inst1->attr10 = inst2->attr10;
	inst1->attr11 = inst2->attr11;
	inst1->attr12 = inst2->attr12;
	inst1->attr13 = inst2->attr13;
	inst1->attr14 = inst2->attr14;
	inst1->attr15 = inst2->attr15;
	inst1->attr16_class = inst2->attr16_class;

}

void constructDataset(STR_INST *dataset) {
	FILE *fp;
	ssize_t read;
	size_t len;
	char *line;
	char** parsedVals;
	int cur;

	fp = fopen(INPUT_DATASET_NAME, "r");
	ifNullExitMessage(fp, "File cannot be opened.");

	len = 0;
	line = NULL;
	cur = 0;

	while ((read = my_getline(&line, &len, fp)) != -1 && cur < NUM_OF_INST) {
		line[strlen(line) - 1] = '\0'; // Eliminate '\n' at the end.
		parsedVals = parseCommaSeperatedValues(line);
		assignStringArrToInstance(cur + 1, &dataset[cur], parsedVals);
		cur++;
		free(parsedVals);
	}
	fclose(fp);

}

STR_INST* createAnInstance(char** instStrArr) { // FOR DEBUG
	STR_INST* anInst;
	anInst = (STR_INST*) malloc(sizeof(STR_INST));

	assignStringArrToInstance(0, anInst, instStrArr);
	return anInst;
}

float calcManDistance(STR_INST *inst1, STR_INST *inst2) {
	float *distArr;
	float sum;
	int i;

	distArr = (float*) malloc(sizeof(float) * (NUM_OF_ATTRS - 1));

	distArr[0] = (inst1->attr1 == inst2->attr1) ? 0 : NOM_DIST_COEFF;
	distArr[1] = ABS(inst1->attr2 - inst2->attr2);
	distArr[2] = ABS((inst1->attr3 - inst2->attr3));
	distArr[3] = (inst1->attr4 == inst2->attr4) ? 0 : NOM_DIST_COEFF;
	distArr[4] = strcmp(inst1->attr5, inst2->attr5) == 0 ? 0 : NOM_DIST_COEFF;
	distArr[5] = strcmp(inst1->attr6, inst2->attr6) == 0 ? 0 : NOM_DIST_COEFF;
	distArr[6] = strcmp(inst1->attr7, inst2->attr7) == 0 ? 0 : NOM_DIST_COEFF;
	distArr[7] = ABS(inst1->attr8 - inst2->attr8);
	distArr[8] = (inst1->attr9 == inst2->attr9) ? 0 : NOM_DIST_COEFF;
	distArr[9] = (inst1->attr10 == inst2->attr10) ? 0 : NOM_DIST_COEFF;
	distArr[10] = ABS(inst1->attr11 - inst2->attr11);
	distArr[11] = (inst1->attr12 == inst2->attr12) ? 0 : NOM_DIST_COEFF;
	distArr[12] = (inst1->attr13 == inst2->attr13) ? 0 : NOM_DIST_COEFF;
	distArr[13] = ABS(inst1->attr14 - inst2->attr14);
	distArr[14] = ABS(inst1->attr15 - inst2->attr15);

	sum = 0;
	for (i = 0; i < NUM_OF_ATTRS - 1; i++)
		sum += distArr[i];

	/*printf("     ");
	 *for (i = 0; i < NUM_OF_ATTRS-1; i++)
	 *		printf("%.2f,", distArr[i]);
	 *	printf("\n");
	 */

	//printf("sum: %f\n", sum);
	free(distArr);
	return (sum / (NUM_OF_ATTRS - 1));

}
STR_INST** splitDatasetInto2(STR_INST* allDS, int testIndexBeg, unsigned int N) {
	int i, j, k;
	STR_INST** splittedDSs; // 0 -> training, 1 -> test

	if (allDS == NULL || (testIndexBeg - 1 + N) > NUM_OF_INST) {

		if (allDS == NULL)
			printf("aaa\n");
		else
			printf("bbb\n");
		printf("Dataset cannot be splitted.\n");
		exit(EXIT_FAIL);
	}

	splittedDSs = (STR_INST**) malloc(sizeof(STR_INST*) * 2);

	splittedDSs[0] = (STR_INST*) malloc(sizeof(STR_INST) * (NUM_OF_INST - N));
	splittedDSs[1] = (STR_INST*) malloc(sizeof(STR_INST) * N);
	for (i = 0; i < testIndexBeg - 1; i++)
		assignInstToInst(&splittedDSs[0][i], &allDS[i]);
	j = i;
	for (k = 0, i = testIndexBeg - 1; i < (testIndexBeg + N - 1); i++, k++)
		assignInstToInst(&splittedDSs[1][k], &allDS[i]);

	for (i = (testIndexBeg + N - 1); i < NUM_OF_INST; i++, j++)
		assignInstToInst(&splittedDSs[0][j], &allDS[i]);

	return splittedDSs;
	printDataSet(splittedDSs[0], 406);
}

void printDistVals_KNN(STR_INST* dataset, STR_INST** SplittedDSs,
		STR_DIST_VAL** distanceVals, int NumOfTestDS) {
	int i, j;
	int numOfTrainingSet = NUM_OF_INST - NumOfTestDS;
/*
	for (i = 0; i < NumOfTestDS; i++) {
		printf("----\n");
		 printf("TestID:%d\n", SplittedDSs[1][i].ID);

		for (j = 0; j < numOfTrainingSet; j++)
			 printf("        TrainingID:%d         <<%d,%f>>       ---->>> %c\n",
					distanceVals[i][j].ID, distanceVals[i][j].ID,
					distanceVals[i][j].dist,
					dataset[distanceVals[i][j].ID - 1].attr16_class);

	}
*/
}

void printDists_KMeans(char* msg, STR_INST* dataset, float* dists) {
	int i;
	printDashes();
	puts(msg);
	for (i = 0; i < NUM_OF_INST; i++)
		printf("\t<<ID:%d,Distance: %f>> \n", i + 1, dists[i]);

}

STR_DIST_VAL** constructDistVals(STR_INST *trainingDS, STR_INST *testDS,
		int numOfTestDS) {
	int i, j;
	int numOfTrainingDS;
	STR_DIST_VAL** distanceVals;

	numOfTrainingDS = NUM_OF_INST - numOfTestDS;

	distanceVals = (STR_DIST_VAL **) malloc(sizeof(STR_DIST_VAL) * numOfTestDS);
	for (i = 0; i < numOfTestDS; i++)
		distanceVals[i] = (STR_DIST_VAL*) malloc(
				sizeof(STR_DIST_VAL) * numOfTrainingDS);

	for (i = 0; i < numOfTestDS; i++)
		for (j = 0; j < numOfTrainingDS; j++) {

			distanceVals[i][j].ID = trainingDS[j].ID;
			distanceVals[i][j].dist = calcManDistance(&trainingDS[j],
					&testDS[i]);
			// printf("********** Test[%d], Training[%d]: %d, %f\n", i, j,
			// 	distanceVals->ID, distanceVals->dist);
		}
	return distanceVals;
}

void swap(STR_DIST_VAL* distVal1, STR_DIST_VAL* distVal2) {

	STR_DIST_VAL* tmpDistVal = (STR_DIST_VAL*) malloc(sizeof(STR_DIST_VAL));

	tmpDistVal->ID = distVal1->ID;
	tmpDistVal->dist = distVal1->dist;

	distVal1->ID = distVal2->ID;
	distVal1->dist = distVal2->dist;

	distVal2->ID = tmpDistVal->ID;
	distVal2->dist = tmpDistVal->dist;

	free(tmpDistVal);
}

int partition(STR_DIST_VAL* distVals, int low, int high) {
	float pivot = distVals[high].dist;
	int i, j;

	i = (low - 1);  // Index of smaller element

	for (j = low; j <= high - 1; j++) {
		// If current element is smaller than or
		// equal to pivot
		if (distVals[j].dist <= pivot) {
			i++;    // increment index of smaller element
			swap(&distVals[i], &distVals[j]);
		}
	}
	swap(&distVals[i + 1], &distVals[high]);
	return (i + 1);
}

void quickSort(STR_DIST_VAL* distVals, int low, int high) {
	if (low < high) {
		/* pi is partitioning index, arr[p] is now
		 at right place */
		int pi = partition(distVals, low, high);

		// Separately sort elements before
		// partition and after partition
		quickSort(distVals, low, pi - 1);
		quickSort(distVals, pi + 1, high);
	}
}

void printDistanceVal(STR_INST** SplittedDSs, STR_DIST_VAL* distanceVals,
		int NumOfTrainingDS) {
	int j;

	for (j = 0; j < NumOfTrainingDS; j++)
		printf("- TrainingID:%d         <<%d,%f>>       \n", distanceVals[j].ID,
				distanceVals[j].ID, distanceVals[j].dist);
}

void sortDistValues(int partSize, STR_DIST_VAL** distanceVals,
		STR_INST **SplittedDSs) {
	int i;

	for (i = 0; i < partSize; i++) {
		quickSort(distanceVals[i], 0, NUM_OF_INST - partSize - 1);
		//printDistanceVal(SplittedDSs, distanceVals[i], NUM_OF_INST - partSize);
		//printDashes();
	}
}

char * func(int kFoldVal, int kVal, int *partSize, int *partIndex,
		STR_INST *dataset) {
	int i, j, k;
	STR_INST **SplittedDSs;
	STR_DIST_VAL** distanceVals;
	int numOfTestSet;
	int charFreq[2];
	char *predictedVals;
	char class_label;

	predictedVals = (char*) malloc(sizeof(char) * NUM_OF_INST);

	for (i = 0; i < kFoldVal; i++) {
		SplittedDSs = splitDatasetInto2(dataset, partIndex[i], partSize[i]); // 0 -> training, 1 -> test
		distanceVals = constructDistVals(SplittedDSs[0], SplittedDSs[1],
				partSize[i]);
		sortDistValues(partSize[i], distanceVals, SplittedDSs);
		printDistVals_KNN(dataset, SplittedDSs, distanceVals, partSize[i]);

		numOfTestSet = partSize[i];
		for (j = 0; j < numOfTestSet; j++) {
			charFreq[0] = 0;
			charFreq[1] = 0;

			printf("Instance %d's closest %d value:\n", SplittedDSs[1][j].ID,
					kVal);

			for (k = 0; k < kVal; k++) {

				class_label = dataset[distanceVals[j][k].ID - 1].attr16_class;

				printf("\t\t%d.order: <<id:%d,distance:%f,class:'%c'>>\n",
						k + 1, distanceVals[j][k].ID, distanceVals[j][k].dist,
						class_label);

				if (class_label == '-')
					charFreq[0] = charFreq[0] + 1;
				else
					charFreq[1] = charFreq[1] + 1;
			}
			predictedVals[SplittedDSs[1][j].ID - 1] =
					charFreq[0] > charFreq[1] ? '-' : '+';
		}

	}

	free(SplittedDSs);
	free(distanceVals);
	return predictedVals;
}

char * calculateKNN(STR_INST *dataset, const int kFoldVal, const int kVal) {
	int i;
	int *partSize;
	int eachPart = NUM_OF_INST / kFoldVal;

	if (NUM_OF_INST < kFoldVal) {
		printf("ERROR: NUM_OF_INST < kFoldVal\n");
		exit(EXIT_FAIL);
	}
	if (kVal <= 0 || kVal % 2 != 1) {
		printf("ERROR: Wrong k value for KNN.\n");
		exit(EXIT_FAIL);
	}

	int *partIndex = (int*) malloc(sizeof(int) * kFoldVal * 2);
	partIndex[0] = 1;
	for (i = 1; i < kFoldVal; i++) {
		partIndex[i] = i * eachPart + 1;
	}

	partSize = (int*) malloc(sizeof(int) * kFoldVal);
	for (i = 0; i < kFoldVal - 1; i++)
		partSize[i] = eachPart;
	partSize[i] = NUM_OF_INST - (kFoldVal - 1) * eachPart;

	printf("Cross Validation Value = %d\n", kFoldVal);
	printf("Part Sizes: ");
	for (i = 0; i < kFoldVal; i++)
		printf("%d ", partSize[i]);
	printf("\n");

	return func(kFoldVal, kVal, partSize, partIndex, dataset);

}

void printBegMsg(char* msg) {
	int i;
	printf("____________________________________________");
	for (i = 0; i < strlen(msg); i++)
		printf("_");
	printf("\n______________________%s______________________\n", msg);
}
void printEndMsg() {
	printf("_________________________________________________\n");
}

void printConfMatrix(int **confMatrix) {
	printf("Confusion Matrix:\n");
	printf("\t Predicted\n");
	printf("\t '+'    '-' \n");
	printf("\t+---------+\n");
	printf("\t|%d    %d|\n", confMatrix[0][0], confMatrix[0][1]);
	printf("\t|%d    %d|\n", confMatrix[1][0], confMatrix[1][1]);
	printf("\t+---------+\n");
}

int** constructConfMatrix(STR_INST* dataset, char *predictedVals) {
	int i;
	const int N = 2;
	int** confMatrix;
	char a, b;

	confMatrix = (int**) malloc(sizeof(int*) * N);
	for (i = 0; i < N; i++)
		confMatrix[i] = (int*) calloc(sizeof(int), N);

	for (i = 0; i < NUM_OF_INST; i++) {
		a = dataset[i].attr16_class;
		b = predictedVals[i];
		if (a == '-') {
			if (b == '-')
				confMatrix[1][1]++;
			else
				confMatrix[1][0]++;
		} else {
			if (b == '-')
				confMatrix[0][1]++;
			else
				confMatrix[0][0]++;
		}

	}

	return confMatrix;

}
void printInstancesLong(char* instsName, STR_INST* insts, int numOfInstances) { // FOR DEBUG ONLY.
	int i;
	printDashes();
	printf("Instance Name: <<%s>>\n", instsName);
	printDashes();
	for (i = 0; i < numOfInstances; i++) {
		printf("\t[%d]:\n", i + 1);
		printf("\t\t ->ID: %d\n", insts[i].ID);
		printf("\t\t ->attr1: <<%c>>\n", insts[i].attr1);
		printf("\t\t ->attr2: <<%f>>\n", insts[i].attr2);
		printf("\t\t ->attr3: <<%f>>\n", insts[i].attr3);
		printf("\t\t ->attr4: <<%c>>\n", insts[i].attr4);
		printf("\t\t ->attr5: <<%s>>\n", insts[i].attr5);
		printf("\t\t ->attr6: <<%s>>\n", insts[i].attr6);
		printf("\t\t ->attr7: <<%s>>\n", insts[i].attr7);
		printf("\t\t ->attr8: <<%f>>\n", insts[i].attr8);
		printf("\t\t ->attr9: <<%c>>\n", insts[i].attr9);
		printf("\t\t ->attr10: <<%c>>\n", insts[i].attr10);
		printf("\t\t ->attr11: <<%f>>\n", insts[i].attr11);
		printf("\t\t ->attr12: <<%c>>\n", insts[i].attr12);
		printf("\t\t ->attr13: <<%c>>\n", insts[i].attr13);
		printf("\t\t ->attr14: <<%f>>\n", insts[i].attr14);
		printf("\t\t ->attr15: <<%f>>\n", insts[i].attr15);
		printf("\t\t ->attr16: <<%c>>\n", insts[i].attr16_class);
	}
	printDashes();
	printf("\n");

}

void printInstances(char* instsName, STR_INST* insts, int numOfInstances) {
	int i;
	printDashes();
	printf("Instance Name: <<%s>>\n", instsName);
	printDashes();
	for (i = 0; i < numOfInstances; i++) {
		printf("\t[%d]: ", i + 1);
		printf(
				"<<%d, %c,%.2f,%.2f,%c,%s,%s,%s,%.2f,%c,%c,%.2f,%c,%c,%.2f,%.2f,%c>>\n",
				insts[i].ID, insts[i].attr1, insts[i].attr2, insts[i].attr3,
				insts[i].attr4, insts[i].attr5, insts[i].attr6, insts[i].attr7,
				insts[i].attr8, insts[i].attr9, insts[i].attr10,
				insts[i].attr11, insts[i].attr12, insts[i].attr13,
				insts[i].attr14, insts[i].attr15, insts[i].attr16_class);
	}
	printDashes();

}

int calcClassLabels(char* instsName, STR_INST* insts, int numOfInstances) {
	int i, posCounter, negCounter;
	posCounter = 0;
	negCounter = 0;
	for (i = 0; i < numOfInstances; i++)
		if (insts[i].attr16_class == '+')
			posCounter++;
		else
			negCounter++;
	printf("\t-> Class Labels:\n\t\t\t1.'+':%d\n\t\t\t2.'-':%d\n", posCounter,
			negCounter);

	printf("\t-> This class's label is:");
	printf(" '%c'\n\n", posCounter > negCounter ? '+' : '-');
	printDashes();

	return MIN(posCounter, negCounter);
}

int** initFreqValue(int numOfAttr, int maxNumOfCategory) {
	int i;
	int** freqVal;
	freqVal = (int**) malloc(sizeof(int*) * numOfAttr);
	for (i = 0; i < numOfAttr; i++)
		freqVal[i] = (int*) calloc(sizeof(int), maxNumOfCategory);
	return freqVal;

}

void calcChFreq(const char *charVals, int *charFreq, int maxChCat,
		char instChar) {
	int i;

	for (i = 0; i < maxChCat; i++) {
		if (charVals[i] == instChar) {
			charFreq[i]++;
			i = maxChCat; // break
		}

	}
}
void calcStrFreq(const char strVals[14][3], int *strFreq, int maxStrCat,
		char* instStr) {
	int i;

	for (i = 0; i < maxStrCat; i++) {
		if (!strcmp(strVals[i], instStr)) {
			strFreq[i]++;
			i = maxStrCat; // break
		}

	}
}

int findMaxIndex(int *vals, int N) {
	int i, maxIndex;

	maxIndex = 0;

	for (i = 1; i < N; i++)
		if (vals[i] > vals[maxIndex])
			maxIndex = i;
	return maxIndex;

}

/* CHAR ATTRIBUTES:
 * 		Attr1 -> charFreq[0], charVals[0]
 * 		Attr4 -> charFreq[1], charVals[1]
 * 		Attr9 -> charFreq[2], charVals[2]
 * 		Attr10 -> charFreq[3], charVals[3]
 * 		Attr12 -> charFreq[4], charVals[4]
 * 		Attr13 -> charFreq[5], charVals[5]
 * CHAR[3] ATTRIBUTES:
 *  	Attr5 ->	strFreq[0], strVals[0]
 *  	Attr6 ->	strFreq[1], strVals[1]
 * 		Attr7 ->	strFreq[2], strVals[2]
 * FLOAT ATTRIBUTES:
 *  	Attr2 ->	strFreq[0]
 *  	Attr3 ->	strFreq[1]
 * 		Attr8 ->	strFreq[2]
 * 		Attr11 ->	strFreq[3]
 * 		Attr14 ->	strFreq[4]
 * 		Attr15 ->	strFreq[5]
 */
STR_INST* getAvgInst(STR_INST* insts, int N) { // N: Num of instances
	int i;
	const int numOfDoubleAttr = 6;
	int numOfCharAttr = 6, numOfStrAttr = 3;
	int maxChCat = 3, maxStrCat = 14;

	int **charFreq, **strFreq;
	double *doubleSum;
	const char charVals[6][3] =
			{ { 'a', 'b', '-' }, { 'l', 'u', 'y' }, { 'f', 't', '-' }, { 'f',
					't', '-' }, { 'f', 't', '-' }, { 'g', 'p', 's' } };
	const char strVals[3][14][3] = { { "g", "gg", "p", "4", "5", "6", "7", "8",
			"9", "10", "11", "12", "13", "14" }, { "aa", "c", "cc", "d", "e",
			"ff", "i", "j", "k", "m", "q", "r", "w", "x" }, { "bb", "dd", "ff",
			"h", "j", "n", "o", "v", "z", "10", "11", "12", "13", "14" } };
	STR_INST* avgInst;

//-------------------------------------------------------------------------

	avgInst = (STR_INST*) malloc(sizeof(STR_INST));
	avgInst->ID = 0;
	avgInst->attr16_class = 'x';

	charFreq = initFreqValue(numOfCharAttr, maxChCat);
	strFreq = initFreqValue(numOfStrAttr, maxStrCat);

	doubleSum = (double*) calloc(sizeof(double), numOfDoubleAttr);

	for (i = 0; i < N; i++) {
		calcChFreq(charVals[0], charFreq[0], maxChCat, insts[i].attr1);
		doubleSum[0] += insts[i].attr2;
		doubleSum[1] += insts[i].attr3;
		calcChFreq(charVals[1], charFreq[1], maxChCat, insts[i].attr4);

		calcStrFreq(strVals[0], strFreq[0], maxStrCat, insts[i].attr5);
		calcStrFreq(strVals[1], strFreq[1], maxStrCat, insts[i].attr6);
		calcStrFreq(strVals[2], strFreq[2], maxStrCat, insts[i].attr7);

		doubleSum[2] += insts[i].attr8;
		calcChFreq(charVals[2], charFreq[2], maxChCat, insts[i].attr9);
		calcChFreq(charVals[3], charFreq[3], maxChCat, insts[i].attr10);
		doubleSum[3] += insts[i].attr11;
		calcChFreq(charVals[4], charFreq[4], maxChCat, insts[i].attr12);
		calcChFreq(charVals[5], charFreq[5], maxChCat, insts[i].attr13);
		doubleSum[4] += insts[i].attr14;
		doubleSum[5] += insts[i].attr15;

	}

	avgInst->attr1 = charVals[0][findMaxIndex(charFreq[0], maxChCat)];
	avgInst->attr2 = (float) doubleSum[0] / N;
	avgInst->attr3 = (float) doubleSum[1] / N;
	avgInst->attr4 = charVals[1][findMaxIndex(charFreq[1], maxChCat)];

	strcpy(avgInst->attr5, strVals[0][findMaxIndex(strFreq[0], maxStrCat)]);
	strcpy(avgInst->attr6, strVals[1][findMaxIndex(strFreq[1], maxStrCat)]);
	strcpy(avgInst->attr7, strVals[2][findMaxIndex(strFreq[2], maxStrCat)]);

	avgInst->attr8 = (float) doubleSum[2] / N;
	avgInst->attr9 = charVals[2][findMaxIndex(charFreq[2], maxChCat)];
	avgInst->attr10 = charVals[3][findMaxIndex(charFreq[3], maxChCat)];
	avgInst->attr11 = (float) doubleSum[3] / N;
	avgInst->attr12 = charVals[4][findMaxIndex(charFreq[4], maxChCat)];
	avgInst->attr13 = charVals[5][findMaxIndex(charFreq[5], maxChCat)];
	avgInst->attr14 = (float) doubleSum[4] / N;
	avgInst->attr15 = (float) doubleSum[5] / N;

	free(charFreq);
	free(strFreq);
	return avgInst;

}

void kMeansCluster(STR_INST *dataset, int iteration) {
	int i, j, k, cur, numOfC1, numOfC2, IncClassifiedInsts;
	STR_INST *seed1, *seed2, *avgInst;
	float *dists1, *dists2;
	STR_INST *class1_DS, *class2_DS;

	seed1 = (STR_INST*) malloc(sizeof(STR_INST));
	seed2 = (STR_INST*) malloc(sizeof(STR_INST));

	dists1 = (float*) calloc(sizeof(float), NUM_OF_INST);
	dists2 = (float*) calloc(sizeof(float), NUM_OF_INST);

	class1_DS = (STR_INST*) malloc(sizeof(STR_INST) * NUM_OF_INST);
	class2_DS = (STR_INST*) malloc(sizeof(STR_INST) * NUM_OF_INST);

	assignInstToInst(seed1, &dataset[rand() % NUM_OF_INST]);
	assignInstToInst(seed2, &dataset[rand() % NUM_OF_INST]);

	printInstances("initial seed1", seed1, 1);
	printInstances("initial seed2", seed2, 1);

	for (cur = 0; cur < iteration; cur++) {

		for (i = 0; i < NUM_OF_INST; i++) {
			dists1[i] = calcManDistance(&dataset[i], seed1);
			dists2[i] = calcManDistance(&dataset[i], seed2);
		}

		for (i = 0, j = 0, k = 0; i < NUM_OF_INST; i++) {
			if (dists1[i] < dists2[i])
				assignInstToInst(&class1_DS[j++], &dataset[i]);
			else
				assignInstToInst(&class2_DS[k++], &dataset[i]);
		}
		numOfC1 = j;
		numOfC2 = k;

		avgInst = getAvgInst(class1_DS, numOfC1);
		assignInstToInst(seed1, avgInst);
		avgInst = getAvgInst(class2_DS, numOfC2);
		assignInstToInst(seed2, avgInst);

	}
	printInstances("Last Seed1", seed1, 1);
	printInstances("Last Seed2", seed2, 1);

	printDists_KMeans("Final Distance Values for Class-1:", dataset, dists1);
	printDists_KMeans("Final Distance Values for Class-2:", dataset, dists2);

	printInstances("class1_DS", class1_DS, numOfC1);
	IncClassifiedInsts = calcClassLabels("class1_DS", class1_DS, numOfC1);

	printInstances("class2_DS", class2_DS, numOfC2);
	IncClassifiedInsts += calcClassLabels("class2_DS", class2_DS, numOfC2);

	printBegMsg("RESULTS:");
	printf("Incorrectly classified instances: %d\n", IncClassifiedInsts);
	printf("Correctly classified instances: %d\n",
	NUM_OF_INST - IncClassifiedInsts);
	printf("Accuracy: %.2f %% \n\n",
			(float) 100 * (NUM_OF_INST - IncClassifiedInsts) / NUM_OF_INST);
	printf("Size of Cluster1: %d\nSize of Cluster2: %d\n", numOfC1, numOfC2);

	free(avgInst);
	free(class1_DS);
	free(class2_DS);
	free(seed1);
	free(seed2);
	free(dists1);
	free(dists2);
}

int main() {
	printBegMsg("BEGINNING OF KNN CALCULATION");
	printf("Distance Measure: Manhattan Distance.\n");

	int i, sum, kfold, kVal;
	char *predictedVals;
	int** confMatrix;
	srand(time(0));

	STR_INST *dataset = (STR_INST*) malloc(sizeof(STR_INST) * NUM_OF_INST);
	ifNullExitMessage(dataset, "Data set cannot be initiated.");

	constructDataset(dataset);

	printBegMsg("DATASET:");
	printDataSet(dataset, NUM_OF_INST);
	printEndMsg();

	printf("Enter Cross Validation Value: ");
	scanf("%d", &kfold);

	printf("Enter k-value of KNN: ");
	scanf("%d", &kVal);

	printBegMsg("MESSAGES:");
	predictedVals = calculateKNN(dataset, kfold, kVal);
	printEndMsg();

	confMatrix = constructConfMatrix(dataset, predictedVals);
	sum = confMatrix[0][0] + confMatrix[1][1];

	printBegMsg("REAL VS PREDICTED:");
	for (i = 0; i < NUM_OF_INST; i++)
		printf("%d. Comparison:\n\tReal\t : '%c'\n\tPredicted: '%c'\n", i + 1,
				dataset[i].attr16_class, predictedVals[i]);

	printBegMsg("RESULTS:");
	printf("Correctly Classified Instances: %d       %f\n", sum,
			(float) sum / NUM_OF_INST);
	printf("Incorrectly Classified Instances: %d       %f\n", NUM_OF_INST - sum,
			(float) (NUM_OF_INST - sum) / NUM_OF_INST);

	printf("Accuracy: %.2f %% \n\n", (float) 100 * sum / NUM_OF_INST);

	printConfMatrix(confMatrix);

	free(predictedVals);

	printBegMsg("END OF KNN CALCULATION");
//----------------------------------------------------------------------
	int iteration;

	printBegMsg("BEGINNING OF K-MEANS CALCULATION");
	printf("Number of Seeds: 2.\n");
	printf("Distance Measure: Manhattan Distance.\n");
	printf("Enter iteration of K-means: ");
	scanf("%d", &iteration);

	kMeansCluster(dataset, iteration);

	free(dataset);
	printBegMsg("END OF K-MEANS CALCULATION");
	puts("Program successfully terminated.");

	return EXIT_SUCCESS;
}

