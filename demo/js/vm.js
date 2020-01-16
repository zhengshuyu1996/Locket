var vm = new Vue({
    el: "#app",
    data: {
        mode: 'welcome',
        show_path: false,
        base_url: 'http://35.189.165.204:80/',
        up_image: null,
        imgData: {
            accept: 'image/gif, image/jpeg, image/png, image/jpg',
        },
        result_path: {
            original: '',
            matting: '',
            transfer: ''
        }
    },
    computed: {
        
    },
    watch: {
        
    },
    methods: {
        add_img(event) {
            let reader = new FileReader();
            let img1 = event.target.files[0];
            let type = img1.type; //文件的类型，判断是否是图片
            let size = img1.size; //文件的大小，判断图片的大小
            if (this.imgData.accept.indexOf(type) == -1) {
                alert('请选择我们支持的图片格式！');
                return false;
            }
            //图片的大小
            if (size > 3145728) {
                alert('请选择3M以内的图片！');
                return false;
            }

            reader.readAsDataURL(img1)
            reader.onload = e => {
                let imgFile = e.target.result;
                let upload_data = {}
                upload_data.image = imgFile
                upload_data.timestamp = new Date().getTime()
                $.ajax({
                    url: this.base_url + 'upload',
                    type: 'POST', 
                    dataType: "json",
                    contentType: "application/json",
                    data: JSON.stringify(upload_data),
                    success: function(res) {
                        console.log('success', res.data)
                        this.result_path = res.data
                        this.mode = 'result'
                        // that.loading = false
                        // that.ai_step = res.data
                        // that.moveAI(res.data)
                    },
                    error: function(err) {
                        // that.loading = false
                        Materialize.toast("Sorry! An error occurs!", 3000)
                        console.log('ajax err', err)
                    }
                })
            }
        }
    },
    mounted () {
        // this.startGame()
        $('.modal').modal();
    }
});