#include <cvzone.h>

SerialData serialData(2, 1); //(numOfValsRec,digitsPerValRec)
int valsRec[2]; // array of int with size numOfValsRec 

#include<Servo.h>
Servo x;
Servo y;
void setup() {
serialData.begin();
x.attach(2);
y.attach(3);
Serial.begin(9600);
x.write(90);
y.write(0);
}

void loop() {
 serialData.Get(valsRec);
if(valsRec[0]==HIGH)
{
  x.write(0);

}
if(valsRec[1]==HIGH)
{
  y.write(90);

}
if(valsRec[1]==HIGH&&valsRec[0]==HIGH)
{
  x.write(0);
  y.write(90);

}
if(valsRec[0]==LOW)
{
     x.write(90);
}
if(valsRec[1]==LOW)
{
     y.write(0);
}
}
