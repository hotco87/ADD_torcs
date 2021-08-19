## Requirements & Installation TORCS
1. install xautomation
```
$ sudo apt install xautomation
```
2. install gym and numpy
```
$ pip install gym
$ pip install numpy
```
3. install gym-TORCS
```
$ sudo apt update
$ sudo apt install git
$ sudo apt install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng-dev libxxf86vm-dev
$ git clone https://github.com/ugo-nama-kun/gym_torcs
$ cd gym_torcs/vtorcs-RL-color
$ sudo ./configure
$ make 
$ sudo make install
$ sudo make datainstall

## 만약 make에서 geometry.cpp파일의 isnan에서 에러가 난다면,
## gym_torcs/vtorcs-RL-color/src/drivers/olethros/geometry.cpp
## line 373: isnan을 std::isnan으로 변경
```
4. plib 1.8.5 설치
```
$ wget http://plib.sourceforge.net/dist/plib-1.8.5.tar.gz
$ tar -xvf plib-1.8.5.tar.gz
$ cd plib-1.8.5/
$ ./configure CFLAGS="-O2 -m64 -fPIC" CPPFLAGS="-O2 -fPIC" CXXFLAGS="-O2 -fPIC" LDFLAGS="-L/usr/lib64"
$ sudo make install
```
5. freeglut 설치
```
$ wget http://prdownloads.sourceforge.net/freeglut/freeglut-2.8.1.tar.gz
$ tar -xvf freeglut-2.8.1.tar.gz
$ cd freeglut-2.8.1/
$ ./configure CFLAGS="-O2 -m64 -fPIC" CPPFLAGS="-O2 -fPIC" CXXFLAGS="-O2 -fPIC" LDFLAGS="-L/usr/lib64"
$ make
$ sudo make install
```

## Running TORCS 
```
$ sudo torcs
```

## Development
- 코드를 짤 땐, gym_torcs폴더에서 하거나 wrraper를 같은 위치에 놔야됨
- python example_experiment.py 로 설치가 잘 됐는지 확인가능
- Torcs에 연결이 안된다면
- 톡스실행 -> RACE -> Quick Race -> Configure Race -> Accept 후 뜨는 화면에서
- scr_server1이 있는지 확인

```
ddpg_collect_data.py : collect data using ddpg agent
cBCQ.py : train BCQ agent
cBCQ_model.py : model of BCQ agent
REM.py : train REM agent
REM_model.py : model of REM agent
utils.py : replay buffer
opennpy.py : for utility. change reward function with given dataset.
evaluate_policy.py : evaluate policy
```

## TODO
- Sensor value 참고: http://xed.ch/help/torcs.html
- Goal:normal CBCQ 보다 잘하면 됨.
```
    1. 버퍼분석 (buffer_original_reward3)
    2. 버퍼 10배로 늘리기
    3. cBCQ_model.py 에서 149 line 수정후 CBCQ 코드 다시 실행
```
