var vm = new Vue({
    el: "#app",
    data: {
        mode: 'welcome',
        loading: false,
        base_url: 'http://35.189.165.204:80/',
        up_image: null,
        timestamp: 0,
        img_width: 375,
        img_height: 513,
        transfer_type: 'simpson',
        img_data: {
            accept: 'image/gif, image/jpeg, image/png, image/jpg',
        },
        all_types: ['painting', 'drawing', 'simpson'],
        result_path: {
            original: '',
            matting: 'data/1579146068210_matting.png',
            transfer: 'data/1579146068210_transfer.png'
        },
        style_imgs: {
            painting: ['img/painting/0048.jpg', 'img/painting/0052.jpg', 'img/painting/0053.jpg', 'img/painting/2167.jpg', 'img/painting/2143.jpg'],
            drawing: ['img/drawing/download (3).jpeg', 'img/drawing/download (1).jpeg', 'img/drawing/i - 55.jpeg', 'img/drawing/i - 192.jpeg', 'img/drawing/i - 160.jpeg'],
            simpson: ['img/simpson/pic_0001.jpg', 'img/simpson/pic_0003.jpg', 'img/simpson/pic_0005.jpg', 'img/simpson/pic_0008.jpg', 'img/simpson/pic_0009.jpg']
        },
        dataset: {
            painting: {
                name: 'Art Images: Painting',
                cite: 'https://www.kaggle.com/thedownhill/art-images-drawings-painting-sculpture-engraving'
            },
            drawing: {
                name: 'Art Images: Drawing',
                cite: 'https://www.kaggle.com/thedownhill/art-images-drawings-painting-sculpture-engraving'
            },
            simpson: {
                name: 'Simpson Faces',
                cite: 'https://www.kaggle.com/alexattia/the-simpsons-characters-dataset'
            }
        }
    },
    computed: {
    },
    watch: {
    },
    methods: {
        addImg(event) {
            let reader = new FileReader();
            let img1 = event.target.files[0];
            let type = img1.type; //文件的类型，判断是否是图片
            let size = img1.size; //文件的大小，判断图片的大小
            if (this.img_data.accept.indexOf(type) == -1) {
                alert('请选择我们支持的图片格式！');
                return false;
            }
            //图片的大小
            if (size > 3145728) {
                alert('请选择3M以内的图片！');
                return false;
            }

            let that = this
            reader.readAsDataURL(img1)
            reader.onload = e => {
                that.up_image = e.target.result;
                // use timestamp as the identifier of the image
                that.timestamp = new Date().getTime()
                // get w & h of the image
                let image_load = new Image();  
                image_load.onload = function(){  
                    that.img_width = image_load.width;
                    that.img_height = image_load.height;
                    // console.log(that.img_width, that.img_height, that.up_image)
                }
                image_load.src = that.up_image;
                that.mode = 'choice'
                // that.uploadImg()
            }
        },
        goBack() {
            if (this.mode == 'result')
                this.mode = 'choice'
            else if (this.mode == 'choice')
                this.mode = 'welcome'
            else if (this.mode == 'style')
                this.mode = 'result'
        },
        restart() {
            this.mode = 'welcome'
        },
        chooseStyle(transfer_type) {
            if (transfer_type == 'random') {
                let index = Math.floor(Math.random()*10) % this.all_types.length
                this.transfer_type = this.all_types[index];
            } else {
                this.transfer_type = transfer_type
            }
            console.log(this.transfer_type)
            this.uploadImg()
        },
        uploadImg() {
            let that = this
            console.log({
                    image: this.up_image,
                    timestamp: this.timestamp,
                    trans_type: this.transfer_type
                })
            // this.loading = true
            // setTimeout(function(){
            //     that.loading = false
            //     that.mode = 'result'
            // }, 5000)
            // return 
            this.loading = true
            $.ajax({
                url: this.base_url + 'upload',
                type: 'POST', 
                dataType: "json",
                contentType: "application/json",
                data: JSON.stringify({
                    image: this.up_image,
                    timestamp: this.timestamp,
                    trans_type: this.transfer_type
                }),
                success: function(res) {
                    console.log('success', res.data)
                    that.result_path = res.data
                    that.mode = 'result'
                    that.loading = false
                },
                error: function(err) {
                    that.loading = false
                    Materialize.toast("Sorry! An error occurs!", 3000)
                    console.log('ajax err', err)
                }
            })
        }
    },
    mounted () {
        // this.startGame()
        $('.slider').slider();
    }
});