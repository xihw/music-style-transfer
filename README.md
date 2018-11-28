# music-style-transfer

Download data from [here](https://s3.us-east-2.amazonaws.com/music-style-transfer/data.zip) and put it under project root path.

## Train on AWS
1. have pem set up locally
2. get ec2 dns name [here](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#Instances:sort=instanceId)
3. `ssh -i dl-ec2.pem ubuntu@ec2-34-229-144-216.compute-1.amazonaws.com`  
   (remember to replace the dns name)
4. `git clone https://github.com/xihw/music-style-transfer.git`  
   and cd to it
5. download data:  
   `curl https://s3.us-east-2.amazonaws.com/music-style-transfer/data.zip --output data.zip`  
   `unzip data.zip`
6. `nohup python main.py &`
   
