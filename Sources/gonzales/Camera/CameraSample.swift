struct CameraSample {

        init(film: TwoRandomVariables = (0, 0), lens: TwoRandomVariables = (0, 0), filterWeight: FloatX = 1) {
                self.film = film
                self.lens = lens
                self.filterWeight = filterWeight
        }

        let film: TwoRandomVariables
        let lens: TwoRandomVariables
        let filterWeight: FloatX
}
